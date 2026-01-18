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
def get_model_specs(model_name):
    """Return a dict with configurations required for configuring `model_name` model."""
    if model_name == 'lm':
        return lm_wikitext2.get_model_config()
    else:
        raise RuntimeError('Unrecognized args.model_mame ' % args.model_name)