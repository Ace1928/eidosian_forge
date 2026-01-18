from collections import defaultdict
import gc
import logging
import math
import time
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
from benchmarks.golden_configs.lm_wikitext2 import Pipe as lm_wikitext2
from fairscale.fair_dev.testing.testing import dist_init
from fairscale.nn import Pipe
from fairscale.nn.model_parallel import initialize_model_parallel
def generate_balance(num_devices, num_layers):
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance