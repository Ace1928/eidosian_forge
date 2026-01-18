import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def demoException(s):
    try:
        Expression.fromstring(s)
    except LogicalExpressionException as e:
        print(f'{e.__class__.__name__}: {e}')