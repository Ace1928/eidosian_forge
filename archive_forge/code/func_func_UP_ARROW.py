import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def func_UP_ARROW(self, modifier):
    if self.focusedIndex > 0:
        self.focusedIndex -= 1
        if self.renderOffset > 0:
            self.renderOffset -= 1
        self.repaint()