from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def is_any(x, *args):
    return any((x is arg for arg in args))