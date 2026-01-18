import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def make_QuanifiedExpression(self, factory, variable, term):
    return factory(variable, term)