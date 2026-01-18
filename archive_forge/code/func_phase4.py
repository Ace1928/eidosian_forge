import os
import stat
from itertools import filterfalse
from types import GenericAlias
def phase4(self):
    self.subdirs = {}
    for x in self.common_dirs:
        a_x = os.path.join(self.left, x)
        b_x = os.path.join(self.right, x)
        self.subdirs[x] = self.__class__(a_x, b_x, self.ignore, self.hide)