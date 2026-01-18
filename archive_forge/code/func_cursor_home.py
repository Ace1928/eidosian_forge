import codecs
import copy
import sys
import warnings
def cursor_home(self, r=1, c=1):
    self.cur_r = r
    self.cur_c = c
    self.cursor_constrain()