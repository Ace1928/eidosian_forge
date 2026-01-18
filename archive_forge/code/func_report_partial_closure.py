import os
import stat
from itertools import filterfalse
from types import GenericAlias
def report_partial_closure(self):
    self.report()
    for sd in self.subdirs.values():
        print()
        sd.report()