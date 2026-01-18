import codecs
import copy
import sys
import warnings
def cursor_constrain(self):
    """This keeps the cursor within the screen area.
        """
    self.cur_r = constrain(self.cur_r, 1, self.rows)
    self.cur_c = constrain(self.cur_c, 1, self.cols)