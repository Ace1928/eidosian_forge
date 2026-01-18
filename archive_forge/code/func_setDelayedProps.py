import sys
from pyside2uic.properties import Properties
from pyside2uic.uiparser import UIParser
from pyside2uic.Compiler import qtproxies
from pyside2uic.Compiler.indenter import createCodeIndenter, getIndenter, \
from pyside2uic.Compiler.qobjectcreator import CompilerCreatorPolicy
from pyside2uic.Compiler.misc import write_import
def setDelayedProps(self):
    write_code('')
    write_code('self.retranslateUi(%s)' % self.toplevelWidget)
    UIParser.setDelayedProps(self)