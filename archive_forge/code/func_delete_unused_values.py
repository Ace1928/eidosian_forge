import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def delete_unused_values(user: Node):
    """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
    if user.op == 'placeholder':
        return
    if user.op == 'output':
        body.append('\n')
        return
    nodes_to_delete = user_to_last_uses.get(user, [])
    if len(nodes_to_delete):
        to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
        body.append(f';  {to_delete_str}\n')
    else:
        body.append('\n')