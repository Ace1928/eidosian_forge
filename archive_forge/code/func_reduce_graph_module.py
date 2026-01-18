import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
@compatibility(is_backward_compatible=True)
def reduce_graph_module(body: Dict[Any, Any], import_block: str) -> torch.nn.Module:
    fn_src = body.get('_code') or body['code']
    forward = _forward_from_src(import_block + fn_src, {})
    return _deserialize_graph_module(forward, body)