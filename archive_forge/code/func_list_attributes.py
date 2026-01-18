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
def list_attributes(self, obj: Any) -> List[str]:
    with inspection.AttrCleaner(obj):
        return dir(obj)