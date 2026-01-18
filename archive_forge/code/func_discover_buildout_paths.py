import os
import re
from pathlib import Path
from importlib.machinery import all_suffixes
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ContextualizedNode
from jedi.inference.helpers import is_string, get_str_or_none
from jedi.parser_utils import get_cached_code_lines
from jedi.file_io import FileIO
from jedi import settings
from jedi import debug
def discover_buildout_paths(inference_state, script_path):
    buildout_script_paths = set()
    for buildout_script_path in _get_buildout_script_paths(script_path):
        for path in _get_paths_from_buildout_script(inference_state, buildout_script_path):
            buildout_script_paths.add(path)
            if len(buildout_script_paths) >= _BUILDOUT_PATH_INSERTION_LIMIT:
                break
    return buildout_script_paths