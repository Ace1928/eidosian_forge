import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def func_PGDN(self, modifier):
    if self.renderOffset != self.height - 1:
        change = self.height - self.renderOffset - 1
        if change + self.focusedIndex >= len(self.sequence):
            change = len(self.sequence) - self.focusedIndex - 1
        self.focusedIndex += change
        self.renderOffset = self.height - 1
    else:
        self.focusedIndex = min(len(self.sequence) - 1, self.focusedIndex + self.height)
    self.repaint()