import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if parent_fqn == '':
        return child_fqn
    parent_split = parent_fqn.split('.')
    child_split = child_fqn.split('.')
    assert child_split[:len(parent_split)] == parent_split, f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'"
    return '.'.join(child_split[len(parent_split):])