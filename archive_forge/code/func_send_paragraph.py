import formatter
import string
from types import *
import htmllib
import piddle
def send_paragraph(self, blankline):
    self.send_line_break()
    self.y = self.y + self.oldLineHeight * blankline