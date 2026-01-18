import glob
import inspect
import pickle
import re
from importlib import import_module
from os import path
from typing import IO, Any, Dict, List, Pattern, Set, Tuple
import sphinx
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import red  # type: ignore
from sphinx.util.inspect import safe_getattr
def ignore_pyobj(self, full_name: str) -> bool:
    for exp in self.py_ignorexps:
        if exp.search(full_name):
            return True
    return False