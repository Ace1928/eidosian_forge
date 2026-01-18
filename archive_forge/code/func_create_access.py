import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def create_access(inference_state, obj):
    return inference_state.compiled_subprocess.get_or_create_access_handle(obj)