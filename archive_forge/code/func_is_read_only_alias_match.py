import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def is_read_only_alias_match(arg, ret):
    shared_aliases = arg.alias_set & ret.alias_set
    return len(shared_aliases) > 0 and (not arg.is_write)