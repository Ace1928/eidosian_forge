import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
def create_load_none():
    return create_instruction('LOAD_CONST', argval=None)