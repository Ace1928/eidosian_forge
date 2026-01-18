import datetime
import inspect
import re
import statistics
from functools import wraps
from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get
@null_if_any('this', 'expression')
def arraytostring(this, expression, null=None):
    return expression.join((x for x in (x if x is not None else null for x in this) if x is not None))