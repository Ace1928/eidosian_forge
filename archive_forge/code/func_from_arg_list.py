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
def from_arg_list(cls, args):
    if cls.is_var_len_args:
        all_arg_keys = list(cls.arg_types)
        non_var_len_arg_keys = all_arg_keys[:-1] if cls.is_var_len_args else all_arg_keys
        num_non_var = len(non_var_len_arg_keys)
        args_dict = {arg_key: arg for arg, arg_key in zip(args, non_var_len_arg_keys)}
        args_dict[all_arg_keys[-1]] = args[num_non_var:]
    else:
        args_dict = {arg_key: arg for arg, arg_key in zip(args, cls.arg_types)}
    return cls(**args_dict)