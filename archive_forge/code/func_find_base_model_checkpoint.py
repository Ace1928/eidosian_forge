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
def find_base_model_checkpoint(model_type: str, model_files: Optional[Dict[str, Union[Path, List[Path]]]]=None) -> str:
    """
    Finds the model checkpoint used in the docstrings for a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        model_files (`Dict[str, Union[Path, List[Path]]`, *optional*):
            The files associated to `model_type`. Can be passed to speed up the function, otherwise will be computed.

    Returns:
        `str`: The checkpoint used.
    """
    if model_files is None:
        model_files = get_model_files(model_type)
    module_files = model_files['model_files']
    for fname in module_files:
        if 'modeling' not in str(fname):
            continue
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
            if _re_checkpoint_for_doc.search(content) is not None:
                checkpoint = _re_checkpoint_for_doc.search(content).groups()[0]
                checkpoint = checkpoint.replace('"', '')
                checkpoint = checkpoint.replace("'", '')
                return checkpoint
    return ''