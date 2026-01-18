import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
def build_return_ann_stmt(outputs):
    return_type_ann = ''
    return_statement_str = 'return '
    if len(outputs) == 0:
        return_type_ann += ' -> None'
    if len(outputs) == 1:
        return_type_ann = ' -> ' + outputs[0].ann
        return_statement_str += outputs[0].name
    if len(outputs) > 1:
        return_type_ann = ' -> Tuple'
        return_type_ann += '[' + ', '.join([var.ann for var in outputs]) + ']'
        return_statement_str += ', '.join([var.name for var in outputs])
    return (return_type_ann, return_statement_str)