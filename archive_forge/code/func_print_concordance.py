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
def print_concordance(self, word, width=80, lines=25):
    """
        Print concordance lines given the query word.
        :param word: The target word or phrase (a list of strings)
        :type word: str or list
        :param lines: The number of lines to display (default=25)
        :type lines: int
        :param width: The width of each line, in characters (default=80)
        :type width: int
        :param save: The option to save the concordance.
        :type save: bool
        """
    concordance_list = self.find_concordance(word, width=width)
    if not concordance_list:
        print('no matches')
    else:
        lines = min(lines, len(concordance_list))
        print(f'Displaying {lines} of {len(concordance_list)} matches:')
        for i, concordance_line in enumerate(concordance_list[:lines]):
            print(concordance_line.line)