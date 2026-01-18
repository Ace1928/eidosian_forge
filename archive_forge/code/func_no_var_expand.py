import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def no_var_expand(magic_func):
    """Mark a magic function as not needing variable expansion

    By default, IPython interprets `{a}` or `$a` in the line passed to magics
    as variables that should be interpolated from the interactive namespace
    before passing the line to the magic function.
    This is not always desirable, e.g. when the magic executes Python code
    (%timeit, %time, etc.).
    Decorate magics with `@no_var_expand` to opt-out of variable expansion.

    .. versionadded:: 7.3
    """
    setattr(magic_func, MAGIC_NO_VAR_EXPAND_ATTR, True)
    return magic_func