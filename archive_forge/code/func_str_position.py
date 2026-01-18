import datetime
import inspect
import re
import statistics
from functools import wraps
from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get
@null_if_any('substr', 'this')
def str_position(substr, this, position=None):
    position = position - 1 if position is not None else position
    return this.find(substr, position) + 1