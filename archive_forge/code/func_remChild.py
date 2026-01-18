import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def remChild(self, child):
    assert child.parent is self
    child.parent = None
    self.children.remove(child)
    self.repaint()