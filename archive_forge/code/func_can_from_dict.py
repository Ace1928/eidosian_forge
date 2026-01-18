import contextlib
import copy
import functools
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import (
import srsly
from .backends import CupyOps, NumpyOps, Ops, ParamServer, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .types import FloatsXd
from .util import (
def can_from_dict(self, msg: Dict, *, strict: bool=True) -> bool:
    """Check whether a dictionary is compatible with the model.
        If 'strict', the function returns False if the model has an attribute
        already loaded that would be changed.
        """
    if 'nodes' not in msg.keys():
        return False
    nodes = list(self.walk())
    if len(msg['nodes']) != len(nodes):
        return False
    for i, node in enumerate(nodes):
        info = msg['nodes'][i]
        if strict and info['name'] != node.name:
            return False
        if len(msg['shims'][i]) != len(node.shims):
            return False
        for dim, value in info['dims'].items():
            has_dim = node.has_dim(dim)
            if has_dim is False:
                return False
            elif has_dim and node.get_dim(dim) != value:
                return False
        for param_name, value in msg['params'][i].items():
            has_param = node.has_param(param_name)
            if has_param is False:
                return False
            elif has_param and value is not None:
                param = node.get_param(param_name)
                if param.shape != value.shape:
                    return False
        if strict:
            for attr, value in msg['attrs'][i].items():
                if attr in node.attrs:
                    try:
                        serialized = serialize_attr(node.attrs[attr], node.attrs[attr], attr, node)
                    except TypeError:
                        continue
                    if serialized != value:
                        return False
    return True