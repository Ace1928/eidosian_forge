from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
def find_python_name(self, name, mark):
    if not name:
        raise ConstructorError('while constructing a Python object', mark, 'expected non-empty name appended to the tag', mark)
    if u'.' in name:
        lname = name.split('.')
        lmodule_name = lname
        lobject_name = []
        while len(lmodule_name) > 1:
            lobject_name.insert(0, lmodule_name.pop())
            module_name = '.'.join(lmodule_name)
            try:
                __import__(module_name)
                break
            except ImportError:
                continue
    else:
        module_name = builtins_module
        lobject_name = [name]
    try:
        __import__(module_name)
    except ImportError as exc:
        raise ConstructorError('while constructing a Python object', mark, 'cannot find module %r (%s)' % (utf8(module_name), exc), mark)
    module = sys.modules[module_name]
    object_name = '.'.join(lobject_name)
    obj = module
    while lobject_name:
        if not hasattr(obj, lobject_name[0]):
            raise ConstructorError('while constructing a Python object', mark, 'cannot find %r in the module %r' % (utf8(object_name), module.__name__), mark)
        obj = getattr(obj, lobject_name.pop(0))
    return obj