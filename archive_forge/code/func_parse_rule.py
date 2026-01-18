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
def parse_rule(self, node):
    type = Rule.TYPE_INCLUDE if node.tag == 'Include' else Rule.TYPE_EXCLUDE
    tree = ast.Expression(lineno=1, col_offset=0)
    expr = self.parse_bool_op(node, ast.Or())
    if expr:
        tree.body = expr
    else:
        tree.body = _ast_const('False')
    ast.fix_missing_locations(tree)
    return Rule(type, tree)