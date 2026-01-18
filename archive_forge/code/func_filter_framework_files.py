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
def filter_framework_files(files: List[Union[str, os.PathLike]], frameworks: Optional[List[str]]=None) -> List[Union[str, os.PathLike]]:
    """
    Filter a list of files to only keep the ones corresponding to a list of frameworks.

    Args:
        files (`List[Union[str, os.PathLike]]`): The list of files to filter.
        frameworks (`List[str]`, *optional*): The list of allowed frameworks.

    Returns:
        `List[Union[str, os.PathLike]]`: The list of filtered files.
    """
    if frameworks is None:
        frameworks = get_default_frameworks()
    framework_to_file = {}
    others = []
    for f in files:
        parts = Path(f).name.split('_')
        if 'modeling' not in parts:
            others.append(f)
            continue
        if 'tf' in parts:
            framework_to_file['tf'] = f
        elif 'flax' in parts:
            framework_to_file['flax'] = f
        else:
            framework_to_file['pt'] = f
    return [framework_to_file[f] for f in frameworks if f in framework_to_file] + others