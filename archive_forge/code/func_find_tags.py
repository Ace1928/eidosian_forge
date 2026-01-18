import re
import tokenize
from collections import OrderedDict
from importlib import import_module
from inspect import Signature
from os import path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile
from sphinx.errors import PycodeError
from sphinx.pycode.parser import Parser
def find_tags(self) -> Dict[str, Tuple[str, int, int]]:
    """Find class, function and method definitions and their location."""
    self.analyze()
    return self.tags