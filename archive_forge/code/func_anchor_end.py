import formatter
import string
from types import *
import htmllib
import piddle
def anchor_end(self):
    if self.anchor:
        self.color = self.oldcolor
        self.anchor = None