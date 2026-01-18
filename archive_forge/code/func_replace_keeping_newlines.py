from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def replace_keeping_newlines(m):
    return ' ' + m.group().count('\n') * '\n'