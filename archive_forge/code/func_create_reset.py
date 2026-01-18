import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
def create_reset():
    return [create_instruction('LOAD_FAST', argval=ctx_name), create_instruction('LOAD_METHOD', argval='__exit__'), create_instruction('LOAD_CONST', argval=None), create_dup_top(), create_dup_top(), *create_call_method(3), create_instruction('POP_TOP')]