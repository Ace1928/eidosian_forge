import sys
from pyside2uic.properties import Properties
from pyside2uic.uiparser import UIParser
from pyside2uic.Compiler import qtproxies
from pyside2uic.Compiler.indenter import createCodeIndenter, getIndenter, \
from pyside2uic.Compiler.qobjectcreator import CompilerCreatorPolicy
from pyside2uic.Compiler.misc import write_import
def compileUi(self, input_stream, output_stream, from_imports):
    createCodeIndenter(output_stream)
    w = self.parse(input_stream)
    indenter = getIndenter()
    indenter.write('')
    self.factory._cpolicy._writeOutImports()
    for res in self._resources:
        write_import(res, from_imports)
    return {'widgetname': str(w), 'uiclass': w.uiclass, 'baseclass': w.baseclass}