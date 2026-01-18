import os
import re
from functools import wraps
from collections import namedtuple
from typing import Dict, Mapping, Tuple
from pathlib import Path
from jedi import settings
from jedi.file_io import FileIO
from jedi.parser_utils import get_cached_code_lines
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.stub_value import TypingModuleWrapper, StubModuleValue
from jedi.inference.value import ModuleValue
def parse_stub_module(inference_state, file_io):
    return inference_state.parse(file_io=file_io, cache=True, diff_cache=settings.fast_parser, cache_path=settings.cache_directory, use_latest_grammar=True)