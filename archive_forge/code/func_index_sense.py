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
def index_sense(self, version=None):
    """Read sense key to synset id mapping from index.sense file in corpus directory"""
    fn = 'index.sense'
    if version:
        from nltk.corpus import CorpusReader, LazyCorpusLoader
        ixreader = LazyCorpusLoader(version, CorpusReader, '.*/' + fn)
    else:
        ixreader = self
    with ixreader.open(fn) as fp:
        sensekey_map = {}
        for line in fp:
            fields = line.strip().split()
            sensekey = fields[0]
            pos = self._pos_names[int(sensekey.split('%')[1].split(':')[0])]
            sensekey_map[sensekey] = f'{fields[1]}-{pos}'
    return sensekey_map