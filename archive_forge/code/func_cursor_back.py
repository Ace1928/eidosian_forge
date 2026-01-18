import codecs
import copy
import sys
import warnings
def cursor_back(self, count=1):
    self.cur_c = self.cur_c - count
    self.cursor_constrain()