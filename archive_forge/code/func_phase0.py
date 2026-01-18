import os
import stat
from itertools import filterfalse
from types import GenericAlias
def phase0(self):
    self.left_list = _filter(os.listdir(self.left), self.hide + self.ignore)
    self.right_list = _filter(os.listdir(self.right), self.hide + self.ignore)
    self.left_list.sort()
    self.right_list.sort()