import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def sizeHint(self):
    if self.minVisible is not None:
        return (self._width, self.minVisible)