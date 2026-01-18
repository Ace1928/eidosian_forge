import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def headerItem(self):
    return QtWidgets.QWidget('%s.headerItem()' % self, False, (), noInstantiation=True)