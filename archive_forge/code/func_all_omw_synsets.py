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
def all_omw_synsets(self, pos=None, lang=None):
    if lang not in self.langs():
        return None
    self._load_lang_data(lang)
    for of in self._lang_data[lang][0]:
        if not pos or of[-1] == pos:
            ss = self.of2ss(of)
            if ss:
                yield ss