from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def get_indent_spaces(self) -> int:
    sourcefor, n = self._indent_spaces_cache
    if sourcefor == self.source:
        assert n is not None
        return n
    n = find_next_indent(self.source[:-1])
    self._indent_spaces_cache = (self.source, n)
    return n