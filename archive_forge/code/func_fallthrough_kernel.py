from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
import sys
def fallthrough_kernel():
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    """
    raise NotImplementedError('fallthrough_kernel() should never be called.')