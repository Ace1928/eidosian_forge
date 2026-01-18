from __future__ import annotations
import ast
import datetime
import os
import re
import sys
from io import BytesIO, TextIOWrapper
import yaml
import yaml.reader
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.yaml import SafeLoader
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.loader import AnsibleLoader
def find_globals(g, tree):
    """Uses AST to find globals in an ast tree"""
    for child in tree:
        if hasattr(child, 'body') and isinstance(child.body, list):
            find_globals(g, child.body)
        elif isinstance(child, (ast.FunctionDef, ast.ClassDef)):
            g.add(child.name)
            continue
        elif isinstance(child, ast.Assign):
            try:
                g.add(child.targets[0].id)
            except (IndexError, AttributeError):
                pass
        elif isinstance(child, ast.Import):
            g.add(child.names[0].name)
        elif isinstance(child, ast.ImportFrom):
            for name in child.names:
                g_name = name.asname or name.name
                if g_name == '*':
                    continue
                g.add(g_name)