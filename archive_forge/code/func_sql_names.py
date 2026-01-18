from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
@classmethod
def sql_names(cls):
    if cls is Func:
        raise NotImplementedError('SQL name is only supported by concrete function implementations')
    if '_sql_names' not in cls.__dict__:
        cls._sql_names = [camel_to_snake_case(cls.__name__)]
    return cls._sql_names