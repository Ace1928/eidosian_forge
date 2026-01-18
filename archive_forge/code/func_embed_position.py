from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def embed_position(pos, docstring):
    if not Options.embed_pos_in_docstring:
        return docstring
    pos_line = u'File: %s (starting at line %s)' % relative_position(pos)
    if docstring is None:
        return EncodedString(pos_line)
    encoding = docstring.encoding
    if encoding is not None:
        try:
            pos_line.encode(encoding)
        except UnicodeEncodeError:
            encoding = None
    if not docstring:
        doc = EncodedString(pos_line)
    else:
        doc = EncodedString(pos_line + u'\n' + docstring)
    doc.encoding = encoding
    return doc