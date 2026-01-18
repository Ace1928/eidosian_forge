import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def horizontalHeader(self):
    return QtWidgets.QHeaderView('%s.horizontalHeader()' % self, False, (), noInstantiation=True)