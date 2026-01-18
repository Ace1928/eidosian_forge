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
def add_model_to_main_init(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, frameworks: Optional[List[str]]=None, with_processing: bool=True):
    """
    Add a model to the main init of Transformers.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        frameworks (`List[str]`, *optional*):
            If specified, only the models implemented in those frameworks will be added.
        with_processsing (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer/feature extractor/processor of the model should also be added to the init or not.
    """
    with open(TRANSFORMERS_PATH / '__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    idx = 0
    new_lines = []
    framework = None
    while idx < len(lines):
        new_framework = False
        if not is_empty_line(lines[idx]) and find_indent(lines[idx]) == 0:
            framework = None
        elif lines[idx].lstrip().startswith('if not is_torch_available'):
            framework = 'pt'
            new_framework = True
        elif lines[idx].lstrip().startswith('if not is_tf_available'):
            framework = 'tf'
            new_framework = True
        elif lines[idx].lstrip().startswith('if not is_flax_available'):
            framework = 'flax'
            new_framework = True
        if new_framework:
            while lines[idx].strip() != 'else:':
                new_lines.append(lines[idx])
                idx += 1
        if framework is not None and frameworks is not None and (framework not in frameworks):
            new_lines.append(lines[idx])
            idx += 1
        elif re.search(f'models.{old_model_patterns.model_lower_cased}( |")', lines[idx]) is not None:
            block = [lines[idx]]
            indent = find_indent(lines[idx])
            idx += 1
            while find_indent(lines[idx]) > indent:
                block.append(lines[idx])
                idx += 1
            if lines[idx].strip() in [')', ']', '],']:
                block.append(lines[idx])
                idx += 1
            block = '\n'.join(block)
            new_lines.append(block)
            add_block = True
            if not with_processing:
                processing_classes = [old_model_patterns.tokenizer_class, old_model_patterns.image_processor_class, old_model_patterns.feature_extractor_class, old_model_patterns.processor_class]
                processing_classes = [c for c in processing_classes if c is not None]
                for processing_class in processing_classes:
                    block = block.replace(f' "{processing_class}",', '')
                    block = block.replace(f', "{processing_class}"', '')
                    block = block.replace(f' {processing_class},', '')
                    block = block.replace(f', {processing_class}', '')
                    if processing_class in block:
                        add_block = False
            if add_block:
                new_lines.append(replace_model_patterns(block, old_model_patterns, new_model_patterns)[0])
        else:
            new_lines.append(lines[idx])
            idx += 1
    with open(TRANSFORMERS_PATH / '__init__.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))