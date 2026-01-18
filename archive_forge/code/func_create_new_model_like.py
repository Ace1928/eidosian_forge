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
def create_new_model_like(model_type: str, new_model_patterns: ModelPatterns, add_copied_from: bool=True, frameworks: Optional[List[str]]=None, old_checkpoint: Optional[str]=None):
    """
    Creates a new model module like a given model of the Transformers library.

    Args:
        model_type (`str`): The model type to duplicate (like "bert" or "gpt2")
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add "Copied from" statements to all classes in the new model modeling files.
        frameworks (`List[str]`, *optional*):
            If passed, will limit the duplicate to the frameworks specified.
        old_checkpoint (`str`, *optional*):
            The name of the base checkpoint for the old model. Should be passed along when it can't be automatically
            recovered from the `model_type`.
    """
    model_info = retrieve_info_for_model(model_type, frameworks=frameworks)
    model_files = model_info['model_files']
    old_model_patterns = model_info['model_patterns']
    if old_checkpoint is not None:
        old_model_patterns.checkpoint = old_checkpoint
    if len(old_model_patterns.checkpoint) == 0:
        raise ValueError('The old model checkpoint could not be recovered from the model type. Please pass it to the `old_checkpoint` argument.')
    keep_old_processing = True
    for processing_attr in ['image_processor_class', 'feature_extractor_class', 'processor_class', 'tokenizer_class']:
        if getattr(old_model_patterns, processing_attr) != getattr(new_model_patterns, processing_attr):
            keep_old_processing = False
    model_classes = model_info['model_classes']
    old_module_name = model_files['module_name']
    module_folder = TRANSFORMERS_PATH / 'models' / new_model_patterns.model_lower_cased
    os.makedirs(module_folder, exist_ok=True)
    files_to_adapt = model_files['model_files']
    if keep_old_processing:
        files_to_adapt = [f for f in files_to_adapt if 'tokenization' not in str(f) and 'processing' not in str(f) and ('feature_extraction' not in str(f)) and ('image_processing' not in str(f))]
    os.makedirs(module_folder, exist_ok=True)
    for module_file in files_to_adapt:
        new_module_name = module_file.name.replace(old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased)
        dest_file = module_folder / new_module_name
        duplicate_module(module_file, old_model_patterns, new_model_patterns, dest_file=dest_file, add_copied_from=add_copied_from and 'modeling' in new_module_name)
    clean_frameworks_in_init(module_folder / '__init__.py', frameworks=frameworks, keep_processing=not keep_old_processing)
    add_content_to_file(TRANSFORMERS_PATH / 'models' / '__init__.py', f'    {new_model_patterns.model_lower_cased},', add_after=f'    {old_module_name},', exact_match=True)
    add_model_to_main_init(old_model_patterns, new_model_patterns, frameworks=frameworks, with_processing=not keep_old_processing)
    files_to_adapt = model_files['test_files']
    if keep_old_processing:
        files_to_adapt = [f for f in files_to_adapt if 'tokenization' not in str(f) and 'processor' not in str(f) and ('feature_extraction' not in str(f)) and ('image_processing' not in str(f))]

    def disable_fx_test(filename: Path) -> bool:
        with open(filename) as fp:
            content = fp.read()
        new_content = re.sub('fx_compatible\\s*=\\s*True', 'fx_compatible = False', content)
        with open(filename, 'w') as fp:
            fp.write(new_content)
        return content != new_content
    disabled_fx_test = False
    tests_folder = REPO_PATH / 'tests' / 'models' / new_model_patterns.model_lower_cased
    os.makedirs(tests_folder, exist_ok=True)
    with open(tests_folder / '__init__.py', 'w'):
        pass
    for test_file in files_to_adapt:
        new_test_file_name = test_file.name.replace(old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased)
        dest_file = test_file.parent.parent / new_model_patterns.model_lower_cased / new_test_file_name
        duplicate_module(test_file, old_model_patterns, new_model_patterns, dest_file=dest_file, add_copied_from=False, attrs_to_remove=['pipeline_model_mapping', 'is_pipeline_test_to_skip'])
        disabled_fx_test = disabled_fx_test | disable_fx_test(dest_file)
    if disabled_fx_test:
        print('The tests for symbolic tracing with torch.fx were disabled, you can add those once symbolic tracing works for your new model.')
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)
    doc_file = REPO_PATH / 'docs' / 'source' / 'en' / 'model_doc' / f'{old_model_patterns.model_type}.md'
    duplicate_doc_file(doc_file, old_model_patterns, new_model_patterns, frameworks=frameworks)
    insert_model_in_doc_toc(old_model_patterns, new_model_patterns)
    if old_model_patterns.model_type == old_model_patterns.checkpoint:
        print(f"The model you picked has the same name for the model type and the checkpoint name ({old_model_patterns.model_type}). As a result, it's possible some places where the new checkpoint should be, you have {new_model_patterns.model_type} instead. You should search for all instances of {new_model_patterns.model_type} in the new files and check they're not badly used as checkpoints.")
    elif old_model_patterns.model_lower_cased == old_model_patterns.checkpoint:
        print(f"The model you picked has the same name for the model type and the checkpoint name ({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new checkpoint should be, you have {new_model_patterns.model_lower_cased} instead. You should search for all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly used as checkpoints.")
    if old_model_patterns.model_type == old_model_patterns.model_lower_cased and new_model_patterns.model_type != new_model_patterns.model_lower_cased:
        print(f"The model you picked has the same name for the model type and the lowercased model name ({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new model type should be, you have {new_model_patterns.model_lower_cased} instead. You should search for all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly used as the model type.")
    if not keep_old_processing and old_model_patterns.tokenizer_class is not None:
        print('The constants at the start of the new tokenizer file created needs to be manually fixed. If your new model has a tokenizer fast, you will also need to manually add the converter in the `SLOW_TO_FAST_CONVERTERS` constant of `convert_slow_tokenizer.py`.')