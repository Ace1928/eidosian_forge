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
def for_file(cls, filename: str, modname: str) -> 'ModuleAnalyzer':
    if ('file', filename) in cls.cache:
        return cls.cache['file', filename]
    try:
        with tokenize.open(filename) as f:
            string = f.read()
        obj = cls(string, modname, filename)
        cls.cache['file', filename] = obj
    except Exception as err:
        if '.egg' + path.sep in filename:
            obj = cls.cache['file', filename] = cls.for_egg(filename, modname)
        else:
            raise PycodeError('error opening %r' % filename, err) from err
    return obj