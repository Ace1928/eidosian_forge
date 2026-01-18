from __future__ import unicode_literals
import unittest
from io import StringIO
import string
from .. import Scanning
from ..Symtab import ModuleScope
from ..TreeFragment import StringParseContext
from ..Errors import init_thread
def make_scanner(self):
    source = Scanning.StringSourceDescriptor('fake code', code)
    buf = StringIO(code)
    context = StringParseContext('fake context')
    scope = ModuleScope('fake_module', None, None)
    return Scanning.PyrexScanner(buf, source, scope=scope, context=context)