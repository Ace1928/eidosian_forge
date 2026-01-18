import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (
def get_class_in_module(class_name: str, module_path: Union[str, os.PathLike]) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    module_path = module_path.replace(os.path.sep, '.')
    module = importlib.import_module(module_path)
    return getattr(module, class_name)