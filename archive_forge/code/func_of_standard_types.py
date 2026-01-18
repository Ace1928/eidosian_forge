from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def of_standard_types(x, *, check_dict_values: bool, deep: bool):
    if is_standard_types(x, check_dict_values=check_dict_values, deep=deep):
        return x
    else:
        raise CannotEval