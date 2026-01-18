from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def handle_matrix(arg):
    n, m, typstr = tuple(map(lambda x: x.strip(), arg.split(',')))
    assert typstr == 'float'
    result = 'PySide2.QtGui.QMatrix{n}x{m}'.format(**locals())
    return eval(result, namespace)