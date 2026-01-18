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
def lu_ids_and_names(self, name=None):
    """
        Uses the LU index, which is much faster than looking up each LU definition
        if only the names and IDs are needed.
        """
    if not self._lu_idx:
        self._buildluindex()
    return {luID: luinfo.name for luID, luinfo in self._lu_idx.items() if luinfo.status not in self._bad_statuses and (name is None or re.search(name, luinfo.name) is not None)}