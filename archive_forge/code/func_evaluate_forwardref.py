import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
def evaluate_forwardref(ref: ForwardRef, globalns: Dict, localns: Dict) -> Any:
    """Evaluate a forward reference."""
    if sys.version_info > (3, 9):
        return ref._evaluate(globalns, localns, frozenset())
    else:
        return ref._evaluate(globalns, localns)