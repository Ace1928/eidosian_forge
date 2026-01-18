import codecs
import copy
import sys
import warnings
def erase_up(self):
    """Erases the screen from the current line up to the top of the
        screen."""
    self.erase_start_of_line()
    self.fill_region(self.cur_r - 1, 1, 1, self.cols)