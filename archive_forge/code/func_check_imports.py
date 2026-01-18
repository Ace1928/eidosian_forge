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
def check_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """
    Check if the current Python environment contains all the libraries that are imported in a file. Will raise if a
    library is missing.

    Args:
        filename (`str` or `os.PathLike`): The module file to check.

    Returns:
        `List[str]`: The list of relative imports in the file.
    """
    imports = get_imports(filename)
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing_packages.append(imp)
    if len(missing_packages) > 0:
        raise ImportError(f'This modeling file requires the following packages that were not found in your environment: {', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`')
    return get_relative_imports(filename)