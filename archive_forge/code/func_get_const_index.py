import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def get_const_index(code_options, val) -> int:
    for i, v in enumerate(code_options['co_consts']):
        if val is v:
            return i
    code_options['co_consts'] += (val,)
    return len(code_options['co_consts']) - 1