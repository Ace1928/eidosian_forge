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
def add_submodule(self, target: str, m: torch.nn.Module) -> bool:
    """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)
        """
    *prefix, field = target.split('.')
    mod: torch.nn.Module = self
    for item in prefix:
        submod = getattr(mod, item, None)
        if submod is None:
            submod = torch.nn.Module()
            setattr(mod, item, submod)
        if not isinstance(submod, torch.nn.Module):
            return False
        mod = submod
    mod.add_module(field, m)
    return True