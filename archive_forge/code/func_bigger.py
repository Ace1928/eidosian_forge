import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def bigger(self):
    self.percent = min(1.0, max(0.0, self.onScroll(+1)))
    self.repaint()