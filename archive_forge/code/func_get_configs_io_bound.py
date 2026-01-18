import torch
from .. import Config, autotune, cdiv, heuristics, jit
from .. import language as tl
from .matmul_perf_model import early_config_prune, estimate_matmul_time
def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1}, num_stages=num_stages, num_warps=num_warps))
                    for split_k in [2, 4, 8, 16]:
                        configs.append(Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k}, num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs