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
def lu_basic(self, fn_luid):
    """
        Returns basic information about the LU whose id is
        ``fn_luid``. This is basically just a wrapper around the
        ``lu()`` function with "subCorpus" info excluded.

        >>> from nltk.corpus import framenet as fn
        >>> lu = PrettyDict(fn.lu_basic(256), breakLines=True)
        >>> # ellipses account for differences between FN 1.5 and 1.7
        >>> lu # doctest: +ELLIPSIS
        {'ID': 256,
         'POS': 'V',
         'URL': 'https://framenet2.icsi.berkeley.edu/fnReports/data/lu/lu256.xml',
         '_type': 'lu',
         'cBy': ...,
         'cDate': '02/08/2001 01:27:50 PST Thu',
         'definition': 'COD: be aware of beforehand; predict.',
         'definitionMarkup': 'COD: be aware of beforehand; predict.',
         'frame': <frame ID=26 name=Expectation>,
         'lemmaID': 15082,
         'lexemes': [{'POS': 'V', 'breakBefore': 'false', 'headword': 'false', 'name': 'foresee', 'order': 1}],
         'name': 'foresee.v',
         'semTypes': [],
         'sentenceCount': {'annotated': ..., 'total': ...},
         'status': 'FN1_Sent'}

        :param fn_luid: The id number of the desired LU
        :type fn_luid: int
        :return: Basic information about the lexical unit
        :rtype: dict
        """
    return self.lu(fn_luid, ignorekeys=['subCorpus', 'exemplars'])