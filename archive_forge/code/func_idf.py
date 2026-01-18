import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
def idf(self, term):
    """The number of texts in the corpus divided by the
        number of texts that the term appears in.
        If a term does not appear in the corpus, 0.0 is returned."""
    idf = self._idf_cache.get(term)
    if idf is None:
        matches = len([True for text in self._texts if term in text])
        if len(self._texts) == 0:
            raise ValueError('IDF undefined for empty document collection')
        idf = log(len(self._texts) / matches) if matches else 0.0
        self._idf_cache[term] = idf
    return idf