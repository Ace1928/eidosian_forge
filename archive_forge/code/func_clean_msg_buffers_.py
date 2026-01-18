from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
def clean_msg_buffers_(self) -> None:
    """Clean outgoing message buffer"""
    while len(self.out_msg_buffer) > 0:
        req, msg = self.out_msg_buffer.pop()
        req.wait()
        msg.set_()