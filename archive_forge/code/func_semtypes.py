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
def semtypes(self):
    """
        Obtain a list of semantic types.

        >>> from nltk.corpus import framenet as fn
        >>> stypes = fn.semtypes()
        >>> len(stypes) in (73, 109) # FN 1.5 and 1.7, resp.
        True
        >>> sorted(stypes[0].keys())
        ['ID', '_type', 'abbrev', 'definition', 'definitionMarkup', 'name', 'rootType', 'subTypes', 'superType']

        :return: A list of all of the semantic types in framenet
        :rtype: list(dict)
        """
    if not self._semtypes:
        self._loadsemtypes()
    return PrettyList((self._semtypes[i] for i in self._semtypes if isinstance(i, int)))