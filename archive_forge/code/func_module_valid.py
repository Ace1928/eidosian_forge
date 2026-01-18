from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
@staticmethod
def module_valid(mod):
    if getattr(mod, '__file__', None) and (not os.path.isdir(mod.__file__)):
        ending = os.path.splitext(mod.__file__)[-1]
        return ending not in ('.py', '.pyc', '.pyo', '.pyi')
    return False