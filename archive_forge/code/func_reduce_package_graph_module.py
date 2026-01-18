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
def reduce_package_graph_module(importer: PackageImporter, body: Dict[Any, Any], generated_module_name: str) -> torch.nn.Module:
    forward = importer.import_module(generated_module_name).forward
    return _deserialize_graph_module(forward, body)