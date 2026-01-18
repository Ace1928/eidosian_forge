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
def frame_relation_types(self):
    """
        Obtain a list of frame relation types.

        >>> from nltk.corpus import framenet as fn
        >>> frts = sorted(fn.frame_relation_types(), key=itemgetter('ID'))
        >>> isinstance(frts, list)
        True
        >>> len(frts) in (9, 10)    # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(frts[0], breakLines=True)
        {'ID': 1,
         '_type': 'framerelationtype',
         'frameRelations': [<Parent=Event -- Inheritance -> Child=Change_of_consistency>, <Parent=Event -- Inheritance -> Child=Rotting>, ...],
         'name': 'Inheritance',
         'subFrameName': 'Child',
         'superFrameName': 'Parent'}

        :return: A list of all of the frame relation types in framenet
        :rtype: list(dict)
        """
    if not self._freltyp_idx:
        self._buildrelationindex()
    return self._freltyp_idx.values()