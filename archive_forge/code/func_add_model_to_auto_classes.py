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
def add_model_to_auto_classes(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, model_classes: Dict[str, List[str]]):
    """
    Add a model to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`Dict[str, List[str]]`): A dictionary framework to list of model classes implemented.
    """
    for filename in AUTO_CLASSES_PATTERNS:
        new_patterns = []
        for pattern in AUTO_CLASSES_PATTERNS[filename]:
            if re.search('any_([a-z]*)_class', pattern) is not None:
                framework = re.search('any_([a-z]*)_class', pattern).groups()[0]
                if framework in model_classes:
                    new_patterns.extend([pattern.replace('{' + f'any_{framework}_class' + '}', cls) for cls in model_classes[framework]])
            elif '{config_class}' in pattern:
                new_patterns.append(pattern.replace('{config_class}', old_model_patterns.config_class))
            elif '{image_processor_class}' in pattern:
                if old_model_patterns.image_processor_class is not None and new_model_patterns.image_processor_class is not None:
                    new_patterns.append(pattern.replace('{image_processor_class}', old_model_patterns.image_processor_class))
            elif '{feature_extractor_class}' in pattern:
                if old_model_patterns.feature_extractor_class is not None and new_model_patterns.feature_extractor_class is not None:
                    new_patterns.append(pattern.replace('{feature_extractor_class}', old_model_patterns.feature_extractor_class))
            elif '{processor_class}' in pattern:
                if old_model_patterns.processor_class is not None and new_model_patterns.processor_class is not None:
                    new_patterns.append(pattern.replace('{processor_class}', old_model_patterns.processor_class))
            else:
                new_patterns.append(pattern)
        for pattern in new_patterns:
            full_name = TRANSFORMERS_PATH / 'models' / 'auto' / filename
            old_model_line = pattern
            new_model_line = pattern
            for attr in ['model_type', 'model_name']:
                old_model_line = old_model_line.replace('{' + attr + '}', getattr(old_model_patterns, attr))
                new_model_line = new_model_line.replace('{' + attr + '}', getattr(new_model_patterns, attr))
            if 'pretrained_archive_map' in pattern:
                old_model_line = old_model_line.replace('{pretrained_archive_map}', f'{old_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP')
                new_model_line = new_model_line.replace('{pretrained_archive_map}', f'{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP')
            new_model_line = new_model_line.replace(old_model_patterns.model_camel_cased, new_model_patterns.model_camel_cased)
            add_content_to_file(full_name, new_model_line, add_after=old_model_line)
    insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns)