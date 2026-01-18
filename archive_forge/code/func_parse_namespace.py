import dataclasses
import inspect
import sys
from typing import Any, Callable, Tuple
import torch
def parse_namespace(qualname: str) -> Tuple[str, str]:
    splits = qualname.split('::')
    if len(splits) != 2:
        raise ValueError(f'Expected `qualname` to be of the form "namespace::name", but got {qualname}. The qualname passed to the torch.library APIs must consist of a namespace and a name, e.g. aten::sin')
    return (splits[0], splits[1])