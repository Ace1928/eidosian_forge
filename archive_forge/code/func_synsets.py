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
def synsets(self, lemma, pos=None, lang='eng', check_exceptions=True):
    """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        If lang is specified, all the synsets associated with the lemma name
        of that language will be returned.
        """
    lemma = lemma.lower()
    if lang == 'eng':
        get_synset = self.synset_from_pos_and_offset
        index = self._lemma_pos_offset_map
        if pos is None:
            pos = POS_LIST
        return [get_synset(p, offset) for p in pos for form in self._morphy(lemma, p, check_exceptions) for offset in index[form].get(p, [])]
    else:
        self._load_lang_data(lang)
        synset_list = []
        if lemma in self._lang_data[lang][1]:
            for l in self._lang_data[lang][1][lemma]:
                if pos is not None and l[-1] != pos:
                    continue
                synset_list.append(self.of2ss(l))
        return synset_list