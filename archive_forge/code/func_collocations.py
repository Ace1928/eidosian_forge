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
def collocations(self, num=20, window_size=2):
    """
        Print collocations derived from the text, ignoring stopwords.

            >>> from nltk.book import text4
            >>> text4.collocations() # doctest: +NORMALIZE_WHITESPACE
            United States; fellow citizens; years ago; four years; Federal
            Government; General Government; American people; Vice President; God
            bless; Chief Justice; one another; fellow Americans; Old World;
            Almighty God; Fellow citizens; Chief Magistrate; every citizen; Indian
            tribes; public debt; foreign nations


        :param num: The maximum number of collocations to print.
        :type num: int
        :param window_size: The number of tokens spanned by a collocation (default=2)
        :type window_size: int
        """
    collocation_strings = [w1 + ' ' + w2 for w1, w2 in self.collocation_list(num, window_size)]
    print(tokenwrap(collocation_strings, separator='; '))