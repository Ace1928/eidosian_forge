import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def printtype(ex):
    print(f'{ex.str()} : {ex.type}')