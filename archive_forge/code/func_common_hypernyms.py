import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def common_hypernyms(self, other):
    """
        Find all synsets that are hypernyms of this synset and the
        other synset.

        :type other: Synset
        :param other: other input synset.
        :return: The synsets that are hypernyms of both synsets.
        """
    if not self._all_hypernyms:
        self._all_hypernyms = {self_synset for self_synsets in self._iter_hypernym_lists() for self_synset in self_synsets}
    if not other._all_hypernyms:
        other._all_hypernyms = {other_synset for other_synsets in other._iter_hypernym_lists() for other_synset in other_synsets}
    return list(self._all_hypernyms.intersection(other._all_hypernyms))