import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setName(self, name):
    s = super(Token, self).setName(name)
    self.errmsg = 'Expected ' + self.name
    return s