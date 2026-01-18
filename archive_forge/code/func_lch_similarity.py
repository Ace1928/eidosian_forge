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
def lch_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
    return synset1.lch_similarity(synset2, verbose, simulate_root)