import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def characterReceived(self, keyID, modifier):
    if keyID == b'\r':
        self.onSelect(self.sequence[self.focusedIndex])