from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_property_decl(s):
    pos = s.position()
    s.next()
    name = p_ident(s)
    doc, body = p_suite_with_docstring(s, Ctx(level='property'), with_doc_only=True)
    return Nodes.PropertyNode(pos, name=name, doc=doc, body=body)