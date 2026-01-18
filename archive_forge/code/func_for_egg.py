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
@classmethod
def for_egg(cls, filename: str, modname: str) -> 'ModuleAnalyzer':
    SEP = re.escape(path.sep)
    eggpath, relpath = re.split('(?<=\\.egg)' + SEP, filename)
    try:
        with ZipFile(eggpath) as egg:
            code = egg.read(relpath).decode()
            return cls.for_string(code, modname, filename)
    except Exception as exc:
        raise PycodeError('error opening %r' % filename, exc) from exc