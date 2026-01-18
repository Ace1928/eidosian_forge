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
def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Add a tokenizer to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    if old_model_patterns.tokenizer_class is None or new_model_patterns.tokenizer_class is None:
        return
    with open(TRANSFORMERS_PATH / 'models' / 'auto' / 'tokenization_auto.py', 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    idx = 0
    while not lines[idx].startswith('    TOKENIZER_MAPPING_NAMES = OrderedDict('):
        idx += 1
    idx += 1
    while not lines[idx].startswith('TOKENIZER_MAPPING = _LazyAutoMapping'):
        if lines[idx].endswith(','):
            block = lines[idx]
        else:
            block = []
            while not lines[idx].startswith('            ),'):
                block.append(lines[idx])
                idx += 1
            block = '\n'.join(block)
        idx += 1
        if f'"{old_model_patterns.model_type}"' in block and old_model_patterns.tokenizer_class in block:
            break
    new_block = block.replace(old_model_patterns.model_type, new_model_patterns.model_type)
    new_block = new_block.replace(old_model_patterns.tokenizer_class, new_model_patterns.tokenizer_class)
    new_lines = lines[:idx] + [new_block] + lines[idx:]
    with open(TRANSFORMERS_PATH / 'models' / 'auto' / 'tokenization_auto.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))