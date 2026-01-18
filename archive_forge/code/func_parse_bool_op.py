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
def parse_bool_op(self, node, operator):
    values = []
    for child in node:
        rule = self.parse_rule_node(child)
        if rule:
            values.append(rule)
    num_values = len(values)
    if num_values > 1:
        return ast.BoolOp(operator, values)
    elif num_values == 1:
        return values[0]
    return None