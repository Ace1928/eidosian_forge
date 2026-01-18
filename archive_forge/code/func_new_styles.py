import formatter
import string
from types import *
import htmllib
import piddle
def new_styles(self, styles):
    self.send_line_break()
    t = 'new_styles(%s)' % repr(styles)
    self.OutputLine(t, 1)