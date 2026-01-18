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
def lemma_count(self, lemma):
    """Return the frequency count for this Lemma"""
    if lemma._lang != 'eng':
        return 0
    if self._key_count_file is None:
        self._key_count_file = self.open('cntlist.rev')
    line = _binary_search_file(self._key_count_file, lemma._key)
    if line:
        return int(line.rsplit(' ', 1)[-1])
    else:
        return 0