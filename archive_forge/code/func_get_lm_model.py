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
def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""
    ninp = config['ninp']
    nhead = config['nhead']
    initrange = config['initrange']
    dropout = config['dropout']
    vocab_size = config['vocab_size']
    nhid = config['nhid']
    ndecoder = config['num_decoder_layers']
    return transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)