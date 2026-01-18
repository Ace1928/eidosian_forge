from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
def parse_in_msg_buffer(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse in-msg buffer and return msg and ps-weight separately"""
    msg = self.in_msg_buffer
    if not self.regular:
        return (msg.narrow(0, 0, len(msg) - 1), msg[-1])
    else:
        return (msg, self.ps_weight * self.peers_per_itr_device)