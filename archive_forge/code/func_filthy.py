import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def filthy(self):
    self._viewport.filthy()
    self._horiz.filthy()
    self._vert.filthy()
    Widget.filthy(self)