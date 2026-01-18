import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def i18n_void_func(name):

    def _printer(self, *args):
        i18n_print('%s.%s(%s)' % (self, name, ', '.join(map(as_string, args))))
    return _printer