import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def get_indiv_vars(e):
    if isinstance(e, IndividualVariableExpression):
        return {e}
    elif isinstance(e, AbstractVariableExpression):
        return set()
    else:
        return e.visit(get_indiv_vars, lambda parts: reduce(operator.or_, parts, set()))