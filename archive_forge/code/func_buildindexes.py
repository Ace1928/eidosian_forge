import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def buildindexes(self):
    """
        Build the internal indexes to make look-ups faster.
        """
    self._buildframeindex()
    self._buildluindex()
    self._buildcorpusindex()
    self._buildrelationindex()