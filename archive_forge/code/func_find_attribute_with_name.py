import ast
import sys
import builtins
from typing import Dict, Any, Optional
from . import line as line_properties
from .inspection import getattr_safe
def find_attribute_with_name(node, name):
    if isinstance(node, ast.Attribute) and node.attr == name:
        return node
    for item in ast.iter_child_nodes(node):
        r = find_attribute_with_name(item, name)
        if r:
            return r