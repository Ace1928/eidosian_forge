import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    """
    Make sure the name is unique (in a module) and can represents an attr.
    """
    name = re.sub('[^0-9a-zA-Z_]+', '_', name)
    if name[0].isdigit():
        name = f'_{name}'
    while hasattr(mod_traced, name):
        match = re.match('(.*)_(\\d+)$', name)
        if match is None:
            name = name + '_1'
        else:
            base, num = match.group(1, 2)
            name = f'{base}_{int(num) + 1}'
    return name