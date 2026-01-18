import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def handle_negation(self, tok, context):
    return self.make_NegatedExpression(self.process_next_expression(Tokens.NOT))