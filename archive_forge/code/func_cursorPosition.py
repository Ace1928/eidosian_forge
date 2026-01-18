import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def cursorPosition(self, x, y):
    return self.terminal.cursorPosition(self.xoff + min(self.width, x), self.yoff + min(self.height, y))