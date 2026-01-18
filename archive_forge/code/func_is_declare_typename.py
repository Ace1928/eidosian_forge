from pythran.passmanager import Transformation
import pythran.metadata as metadata
from pythran.spec import parse_pytypes
from pythran.types.conversion import pytype_to_ctype
from pythran.utils import isstr
from gast import AST
import gast as ast
import re
def is_declare_typename(s, offset, bounds):
    start = s.rfind(':', 0, offset - 1)
    stop = s.rfind(':', offset + 1)
    if start > 0 and stop > 0:
        bounds.extend((start + 1, stop))
        return True
    else:
        return False