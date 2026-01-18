import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def repaint(self):
    if self._paintCall is None:
        self._paintCall = object()
        self.scheduler(self._paint)
    ContainerWidget.repaint(self)