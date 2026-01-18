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
def collocation_list(self, num=20, window_size=2):
    """
        Return collocations derived from the text, ignoring stopwords.

            >>> from nltk.book import text4
            >>> text4.collocation_list()[:2]
            [('United', 'States'), ('fellow', 'citizens')]

        :param num: The maximum number of collocations to return.
        :type num: int
        :param window_size: The number of tokens spanned by a collocation (default=2)
        :type window_size: int
        :rtype: list(tuple(str, str))
        """
    if not ('_collocations' in self.__dict__ and self._num == num and (self._window_size == window_size)):
        self._num = num
        self._window_size = window_size
        from nltk.corpus import stopwords
        ignored_words = stopwords.words('english')
        finder = BigramCollocationFinder.from_words(self.tokens, window_size)
        finder.apply_freq_filter(2)
        finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
        bigram_measures = BigramAssocMeasures()
        self._collocations = list(finder.nbest(bigram_measures.likelihood_ratio, num))
    return self._collocations