from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def recalculate_mem_size(self):
    self.used_mem_bytes = 0
    for node in self.nodes:
        self.used_mem_bytes += get_extra_size_of(node, self.nodes)