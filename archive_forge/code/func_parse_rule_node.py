from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def parse_rule_node(self, node):
    tag = node.tag
    if tag == 'Or':
        return self.parse_bool_op(node, ast.Or())
    elif tag == 'And':
        return self.parse_bool_op(node, ast.And())
    elif tag == 'Not':
        expr = self.parse_bool_op(node, ast.Or())
        return ast.UnaryOp(ast.Not(), expr) if expr else None
    elif tag == 'All':
        return _ast_const('True')
    elif tag == 'Category':
        category = node.text
        return ast.Compare(left=ast.Str(category), ops=[ast.In()], comparators=[ast.Attribute(value=ast.Name(id='menuentry', ctx=ast.Load()), attr='Categories', ctx=ast.Load())])
    elif tag == 'Filename':
        filename = node.text
        return ast.Compare(left=ast.Str(filename), ops=[ast.Eq()], comparators=[ast.Attribute(value=ast.Name(id='menuentry', ctx=ast.Load()), attr='DesktopFileID', ctx=ast.Load())])