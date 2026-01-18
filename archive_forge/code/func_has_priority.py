import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def has_priority(self, operation, context):
    return self.operator_precedence[operation] < self.operator_precedence[context] or (operation in self.right_associated_operations and self.operator_precedence[operation] == self.operator_precedence[context])