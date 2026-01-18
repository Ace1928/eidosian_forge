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
def similar_words(self, word, n=20):
    scores = defaultdict(int)
    for c in self._word_to_contexts[self._key(word)]:
        for w in self._context_to_words[c]:
            if w != word:
                scores[w] += self._context_to_words[c][word] * self._context_to_words[c][w]
    return sorted(scores, key=scores.get, reverse=True)[:n]