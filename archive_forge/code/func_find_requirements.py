from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
def find_requirements(code: str) -> List[str]:
    """
    Finds the packages required in a string of code.

    Parameters
    ----------
    code : str
       the Python code to run.

    Returns
    -------
    ``List[str]``
        A list of package names that are to be installed for the code to be able to run.

    Examples
    --------
    >>> code = "import numpy as np; import scipy.stats"
    >>> find_imports(code)
    ['numpy', 'scipy']
    """
    code = dedent(code)
    mod = ast.parse(code)
    imports = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for name in node.names:
                node_name = name.name
                imports.add(node_name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name is None:
                continue
            imports.add(module_name.split('.')[0])
    packages = []
    for pkg in sorted(imports):
        pkg = _PACKAGE_MAP.get(pkg, pkg)
        if pkg in _STDLIBS:
            continue
        elif isinstance(pkg, list):
            packages.extend(pkg)
        else:
            packages.append(pkg)
    if any((pdd in code for pdd in _PANDAS_AUTODETECT)) and 'pandas' not in packages:
        packages.append('pandas')
    return [pkg for pkg in packages if pkg not in _IGNORED_PKGS]