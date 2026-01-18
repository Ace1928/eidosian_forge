import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def fd_id(fd):
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)