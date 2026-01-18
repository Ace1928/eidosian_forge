import ast
import logging
import os
import re
import sys
import warnings
from typing import List
from importlib import util
from importlib.metadata import version
from pathlib import Path
from . import Nuitka, run_command
def pyside_imports(py_file: Path):
    modules = []
    contents = py_file.read_text(encoding='utf-8')
    try:
        tree = ast.parse(contents)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                main_mod_name = node.module
                if main_mod_name.startswith('PySide6'):
                    if main_mod_name == 'PySide6':
                        for imported_module in node.names:
                            full_mod_name = imported_module.name
                            if full_mod_name.startswith('Qt'):
                                modules.append(full_mod_name[2:])
                        continue
                    match = mod_pattern.search(main_mod_name)
                    if match:
                        mod_name = match.group('mod_name')
                        modules.append(mod_name)
                    else:
                        logging.warning(f'[DEPLOY] Unable to find module name from{ast.dump(node)}')
            if isinstance(node, ast.Import):
                for imported_module in node.names:
                    full_mod_name = imported_module.name
                    if full_mod_name == 'PySide6':
                        logging.warning(IMPORT_WARNING_PYSIDE.format(str(py_file)))
    except Exception as e:
        raise RuntimeError(f'[DEPLOY] Finding module import failed on file {str(py_file)} with error {e}')
    return set(modules)