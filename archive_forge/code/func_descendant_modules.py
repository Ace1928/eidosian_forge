import torch
from ..modules import Module
from . import comm
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union, cast
from torch._utils import _get_device_index
from collections import OrderedDict
def descendant_modules(module: Module) -> Iterator[Module]:
    gen = module.modules()
    next(gen)
    return gen