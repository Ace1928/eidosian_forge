from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def ast_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    else:
        return None