from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
def head_index(self):
    """
        :return: An value indexing the head of the entire ``DependencySpan``.
        :rtype: int
        """
    return self._head_index