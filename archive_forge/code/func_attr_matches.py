import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
def attr_matches(self, text: str, namespace: Dict[str, Any]) -> Iterator[str]:
    """Taken from rlcompleter.py and bent to my will."""
    m = self.attr_matches_re.match(text)
    if not m:
        return (_ for _ in ())
    expr, attr = m.group(1, 3)
    if expr.isdigit():
        return (_ for _ in ())
    try:
        obj = safe_eval(expr, namespace)
    except EvaluationError:
        return (_ for _ in ())
    return self.attr_lookup(obj, expr, attr)