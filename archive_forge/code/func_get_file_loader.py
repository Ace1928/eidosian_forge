import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
def get_file_loader(path: pathlib.Path):
    """
    Get the file loader for the path.
    """
    return next((lazy_import(loader) for loader, extensions in _file_loader_extensions.items() if path.suffix in extensions), None)