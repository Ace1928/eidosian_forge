import codecs
import copy
import sys
import warnings
def cursor_force_position(self, r, c):
    """Identical to Cursor Home."""
    self.cursor_home(r, c)