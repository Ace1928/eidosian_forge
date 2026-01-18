import codecs
import copy
import sys
import warnings
def cursor_save_attrs(self):
    """Save current cursor position."""
    self.cur_saved_r = self.cur_r
    self.cur_saved_c = self.cur_c