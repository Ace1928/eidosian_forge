import logging
import math
import time
from golden_configs.lm_wikitext2 import MOE as MOEConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
def benchmark_single_process(config_class, args):
    """Benchmark a given model using a single process and multiple devices."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert world_size > 0
    benchmark_config = utils.create_benchmark_config(args.model_name, config_class)
    model_specs = utils.get_model_specs(args.model_name, config_class)
    mp.spawn(train, args=(world_size, benchmark_config, model_specs, args), nprocs=world_size, join=True)