from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def is_standard_types(x, *, check_dict_values: bool, deep: bool):
    try:
        return _is_standard_types_deep(x, check_dict_values, deep)[0]
    except RecursionError:
        return False