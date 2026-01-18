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
def remove_attributes(obj, target_attr):
    """Remove `target_attr` in `obj`."""
    lines = obj.split(os.linesep)
    target_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(f'{target_attr} = '):
            target_idx = idx
            break
        elif line.lstrip().startswith(f'def {target_attr}('):
            target_idx = idx
            break
    if target_idx is None:
        return obj
    line = lines[target_idx]
    indent_level = find_indent(line)
    parsed = extract_block('\n'.join(lines[target_idx:]), indent_level)
    num_lines = len(parsed.split('\n'))
    for idx in range(num_lines):
        lines[target_idx + idx] = None
    for idx in range(target_idx - 1, -1, -1):
        line = lines[idx]
        if (line.lstrip().startswith('#') or line.lstrip().startswith('@')) and find_indent(line) == indent_level:
            lines[idx] = None
        else:
            break
    new_obj = os.linesep.join([x for x in lines if x is not None])
    return new_obj