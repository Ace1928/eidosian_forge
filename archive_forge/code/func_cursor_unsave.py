import codecs
import copy
import sys
import warnings
def cursor_unsave(self):
    """Restores cursor position after a Save Cursor."""
    self.cursor_restore_attrs()