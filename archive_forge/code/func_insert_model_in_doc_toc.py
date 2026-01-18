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
def insert_model_in_doc_toc(old_model_patterns, new_model_patterns):
    """
    Insert the new model in the doc TOC, in the same section as the old model.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    toc_file = REPO_PATH / 'docs' / 'source' / 'en' / '_toctree.yml'
    with open(toc_file, 'r', encoding='utf8') as f:
        content = yaml.safe_load(f)
    api_idx = 0
    while content[api_idx]['title'] != 'API':
        api_idx += 1
    api_doc = content[api_idx]['sections']
    model_idx = 0
    while api_doc[model_idx]['title'] != 'Models':
        model_idx += 1
    model_doc = api_doc[model_idx]['sections']
    old_model_type = old_model_patterns.model_type
    section_idx = 0
    while section_idx < len(model_doc):
        sections = [entry['local'] for entry in model_doc[section_idx]['sections']]
        if f'model_doc/{old_model_type}' in sections:
            break
        section_idx += 1
    if section_idx == len(model_doc):
        old_model = old_model_patterns.model_name
        new_model = new_model_patterns.model_name
        print(f'Did not find {old_model} in the table of content, so you will need to add {new_model} manually.')
        return
    toc_entry = {'local': f'model_doc/{new_model_patterns.model_type}', 'title': new_model_patterns.model_name}
    model_doc[section_idx]['sections'].append(toc_entry)
    model_doc[section_idx]['sections'] = sorted(model_doc[section_idx]['sections'], key=lambda s: s['title'].lower())
    api_doc[model_idx]['sections'] = model_doc
    content[api_idx]['sections'] = api_doc
    with open(toc_file, 'w', encoding='utf-8') as f:
        f.write(yaml.dump(content, allow_unicode=True))