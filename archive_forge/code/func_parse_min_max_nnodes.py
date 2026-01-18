import logging
import os
import sys
import uuid
from argparse import REMAINDER, ArgumentParser
from typing import Callable, List, Tuple, Union
import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.utils.backend_registration import _get_custom_mod_func
def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(':')
    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')
    return (min_nodes, max_nodes)