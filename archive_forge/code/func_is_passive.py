from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
def is_passive(self, rank: Optional[int]=None) -> bool:
    return False