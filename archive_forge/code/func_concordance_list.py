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
def concordance_list(self, word, width=79, lines=25):
    """
        Generate a concordance for ``word`` with the specified context window.
        Word matching is not case-sensitive.

        :param word: The target word or phrase (a list of strings)
        :type word: str or list
        :param width: The width of each line, in characters (default=80)
        :type width: int
        :param lines: The number of lines to display (default=25)
        :type lines: int

        :seealso: ``ConcordanceIndex``
        """
    if '_concordance_index' not in self.__dict__:
        self._concordance_index = ConcordanceIndex(self.tokens, key=lambda s: s.lower())
    return self._concordance_index.find_concordance(word, width)[:lines]