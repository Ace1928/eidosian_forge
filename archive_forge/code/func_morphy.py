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
def morphy(self, form, pos=None, check_exceptions=True):
    """
        Find a possible base form for the given form, with the given
        part of speech, by checking WordNet's list of exceptional
        forms, and by recursively stripping affixes for this part of
        speech until a form in WordNet is found.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        >>> print(wn.morphy('abaci'))
        abacus
        >>> wn.morphy('hardrock', wn.ADV)
        >>> print(wn.morphy('book', wn.NOUN))
        book
        >>> wn.morphy('book', wn.ADJ)
        """
    if pos is None:
        morphy = self._morphy
        analyses = chain((a for p in POS_LIST for a in morphy(form, p)))
    else:
        analyses = self._morphy(form, pos, check_exceptions)
    first = list(islice(analyses, 1))
    if len(first) == 1:
        return first[0]
    else:
        return None