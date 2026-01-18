import heapq
import torch
from .. import cdiv
from .._C.libtriton.triton import runtime
from ..runtime import driver
from ..testing import (get_dram_gbps, get_max_simd_tflops, get_max_tensorcore_tflops, nvsmi)
def early_config_prune(configs, named_args):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    dtsize = named_args['A'].element_size()
    dtype = named_args['A'].dtype
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], config.num_stages)
        max_shared_memory = driver.utils.get_device_properties(device)['max_shared_mem']
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs
    if dtype not in [torch.float16, torch.float32]:
        configs = [config for config in configs if config.kwargs['SPLIT_K'] == 1]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = (kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], kw['SPLIT_K'], config.num_warps, config.num_stages)
        key = (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]
    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps = k
        if capability[0] >= 8:
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8
            ldgsts_latency = 300
            optimal_num_stages = ldgsts_latency / mma_cycles
            nearest = heapq.nsmallest(2, v, key=lambda x: 10 + abs(x[1] - optimal_num_stages) if x[1] - optimal_num_stages < 0 else x[1] - optimal_num_stages)
            for n in nearest:
                pruned_configs.append(n[0])
        else:
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs