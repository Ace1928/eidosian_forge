import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def streamline(self):
    if not self.streamlined:
        self.streamlined = True
        if self.expr is not None:
            self.expr.streamline()
    return self