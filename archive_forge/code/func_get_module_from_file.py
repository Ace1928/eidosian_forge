import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def get_module_from_file(module_file: Union[str, os.PathLike]) -> str:
    """
    Returns the module name corresponding to a module file.
    """
    full_module_path = Path(module_file).absolute()
    module_parts = full_module_path.with_suffix('').parts
    idx = len(module_parts) - 1
    while idx >= 0 and module_parts[idx] != 'transformers':
        idx -= 1
    if idx < 0:
        raise ValueError(f'{module_file} is not a transformers module.')
    return '.'.join(module_parts[idx:])