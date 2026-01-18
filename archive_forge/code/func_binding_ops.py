import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def binding_ops():
    """
    Binding operators
    """
    names = ['existential', 'universal', 'lambda']
    for pair in zip(names, [Tokens.EXISTS, Tokens.ALL, Tokens.LAMBDA, Tokens.IOTA]):
        print('%-15s\t%s' % pair)