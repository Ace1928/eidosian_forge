from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple
import torch
from ... import _is_triton_available
from ..common import register_operator
from .attn_bias import LowerTriangularMask
from .common import (
def import_module_from_path(path: str) -> types.ModuleType:
    """Import a module from the given path, w/o __init__.py"""
    module_path = pathlib.Path(path).resolve()
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert isinstance(spec, importlib.machinery.ModuleSpec)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
    return module