import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def find_device_based_on_size(node) -> Device:
    """Given a node, this function is to find a logical device
            that could fit the node.
            """
    mem_size_needed = get_extra_size_of(node, set())
    device = Device('', -1, -1)
    for d in self.devices:
        if d not in occupied_devices and d.available_mem_bytes >= mem_size_needed:
            device = d
            break
    if device.available_mem_bytes < 0:
        raise RuntimeError(str(node) + 'is too large to fit any device')
    occupied_devices.append(device)
    return device