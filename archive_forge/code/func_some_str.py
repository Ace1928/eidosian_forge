import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def some_str(value):
    try:
        return str(value)
    except:
        return '<unprintable %s object>' % type(value).__name__