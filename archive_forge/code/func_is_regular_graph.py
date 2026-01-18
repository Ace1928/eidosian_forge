from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
def is_regular_graph(self) -> bool:
    return True