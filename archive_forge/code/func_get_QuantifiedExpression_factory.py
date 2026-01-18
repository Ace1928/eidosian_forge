import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def get_QuantifiedExpression_factory(self, tok):
    """This method serves as a hook for other logic parsers that
        have different quantifiers"""
    if tok in Tokens.EXISTS_LIST:
        return ExistsExpression
    elif tok in Tokens.ALL_LIST:
        return AllExpression
    elif tok in Tokens.IOTA_LIST:
        return IotaExpression
    else:
        self.assertToken(tok, Tokens.QUANTS)