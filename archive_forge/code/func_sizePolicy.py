import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def sizePolicy(self):
    sp = LiteralProxyClass()
    sp._uic_name = '%s.sizePolicy()' % self
    return sp