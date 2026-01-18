from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
def mix_out_msg_(self, out_msg: torch.Tensor, ps_weight: torch.Tensor) -> Iterator[torch.Tensor]:
    """Returns a generator mixing messages on the fly"""
    self.refresh_mixing_weights_(residual_adjusted=True)
    self.ps_weight = ps_weight
    if not self.regular:
        out_msg = torch.cat([out_msg, cast(torch.Tensor, self.ps_weight.type(out_msg.dtype))])
    if self._mixing_manager.is_uniform():
        weight = self.mixing_weights['uniform']
        out_msg *= weight.type(out_msg.dtype)
        for _ in self.out_edges:
            yield out_msg
    else:
        for out_edge in self.out_edges:
            weight = self.mixing_weights[out_edge.dest]
            yield out_msg.mul(weight.type(out_msg.dtype))