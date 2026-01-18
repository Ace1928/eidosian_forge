import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
@classmethod
def get_names_from_expression(cls, node):
    """
        Look for all the symbols used in an ast node.
        """
    if isinstance(node, ast.Name):
        yield node.id
    if isinstance(node, (ast.JoinedStr, ast.BoolOp)):
        for n in node.values:
            if isinstance(n, ast.Str):
                yield from cls.get_names_from_expression(n.s)
            else:
                yield from cls.get_names_from_expression(n.value)
    if isinstance(node, ast.BinOp):
        yield from cls.get_names_from_expression(node.right)
        yield from cls.get_names_from_expression(node.left)
    if isinstance(node, ast.IfExp):
        yield from cls.get_names_from_expression(node.test)
        yield from cls.get_names_from_expression(node.body)
        yield from cls.get_names_from_expression(node.orelse)
    if isinstance(node, ast.Subscript):
        yield from cls.get_names_from_expression(node.value)
        yield from cls.get_names_from_expression(node.slice)
    if isinstance(node, ast.Slice):
        yield from cls.get_names_from_expression(node.lower)
        yield from cls.get_names_from_expression(node.upper)
        yield from cls.get_names_from_expression(node.step)
    if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
        for g in node.generators:
            yield from cls.get_names_from_expression(g.iter)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            yield from cls.get_names_from_expression(elt)
    if isinstance(node, ast.Dict):
        for val in node.values:
            yield from cls.get_names_from_expression(val)
    if isinstance(node, ast.UnaryOp):
        yield from cls.get_names_from_expression(node.operand)
    if isinstance(node, ast.comprehension):
        yield from cls.get_names_from_expression(node.iter.value)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            yield f'{node.value.id}.{node.attr}'
    if isinstance(node, ast.Call):
        yield from cls.get_names_from_expression(node.func)
        for arg in node.args:
            yield from cls.get_names_from_expression(arg)
        for keyword in node.keywords:
            yield from cls.get_names_from_expression(keyword.value)