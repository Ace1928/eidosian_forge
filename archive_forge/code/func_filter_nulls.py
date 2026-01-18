import datetime
import inspect
import re
import statistics
from functools import wraps
from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get
def filter_nulls(func, empty_null=True):

    @wraps(func)
    def _func(values):
        filtered = tuple((v for v in values if v is not None))
        if not filtered and empty_null:
            return None
        return func(filtered)
    return _func