import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import time
from datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from models import transformer_lm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from benchmarks.golden_configs.lm_wikitext2 import FSDP as lm_wikitext2
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
def benchmark_fsdp(rank, args, world_size):
    """Benchmark a given model using a single process and multiple devices."""
    init_method_pgroup = 'tcp://localhost:{}'.format(RPC_PORT)
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method=init_method_pgroup)
    torch.cuda.set_device(rank)
    init_random_seed(0)
    logging.basicConfig(level=logging.DEBUG)
    benchmark_config = create_benchmark_config(args.model_name)
    model_specs = get_model_specs(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
    model = model_config['model']
    config = {}
    if args.full_fp16:
        config['compute_dtype'] = torch.float16
        config['mixed_precision'] = False
    if args.enable_auto_wrap:
        with enable_wrap(wrapper_cls=FSDP, **config):
            fsdp_model = auto_wrap(model, auto_wrap_policy=default_auto_wrap_policy)
            fsdp_model = FSDP(fsdp_model, **config)
    else:
        fsdp_model = FSDP(model, **config)
    if args.full_fp16:
        fsdp_model = fsdp_model.half()
    print(f'param dtype {[p.dtype for p in fsdp_model.parameters()]}')
    if args.dry_run:
        train(model_config, fsdp_model, benchmark_config, model_specs, args)
    else:
        benchmark_language_model(model_config, fsdp_model, benchmark_config, model_specs, args)