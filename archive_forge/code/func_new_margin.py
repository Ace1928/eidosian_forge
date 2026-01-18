import formatter
import string
from types import *
import htmllib
import piddle
def new_margin(self, margin, level):
    self.send_line_break()
    self.indent = self.x = self.lmargin + self.indentSize * level