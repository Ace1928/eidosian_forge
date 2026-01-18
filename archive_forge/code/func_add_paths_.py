from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union
from torch import Tensor, nn
def add_paths_(module: nn.Module, prefix: str='') -> None:
    if isinstance(module, search_class):
        paths.append((prefix, module))
    for name, child in module.named_children():
        add_paths_(child, prefix + name + '.')