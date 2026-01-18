import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def findtype(self, variable):
    """:see Expression.findtype()"""
    assert isinstance(variable, Variable), '%s is not a Variable' % variable
    f = self.first.findtype(variable)
    s = self.second.findtype(variable)
    if f == s or s == ANY_TYPE:
        return f
    elif f == ANY_TYPE:
        return s
    else:
        return ANY_TYPE