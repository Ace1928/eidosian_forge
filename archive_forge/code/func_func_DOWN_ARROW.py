import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def func_DOWN_ARROW(self, modifier):
    if self.focusedIndex < len(self.sequence) - 1:
        self.focusedIndex += 1
        if self.renderOffset < self.height - 1:
            self.renderOffset += 1
        self.repaint()