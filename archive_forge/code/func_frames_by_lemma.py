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
def frames_by_lemma(self, pat):
    """
        Returns a list of all frames that contain LUs in which the
        ``name`` attribute of the LU matches the given regular expression
        ``pat``. Note that LU names are composed of "lemma.POS", where
        the "lemma" part can be made up of either a single lexeme
        (e.g. 'run') or multiple lexemes (e.g. 'a little').

        Note: if you are going to be doing a lot of this type of
        searching, you'd want to build an index that maps from lemmas to
        frames because each time frames_by_lemma() is called, it has to
        search through ALL of the frame XML files in the db.

        >>> from nltk.corpus import framenet as fn
        >>> from nltk.corpus.reader.framenet import PrettyList
        >>> PrettyList(sorted(fn.frames_by_lemma(r'(?i)a little'), key=itemgetter('ID'))) # doctest: +ELLIPSIS
        [<frame ID=189 name=Quanti...>, <frame ID=2001 name=Degree>]

        :return: A list of frame objects.
        :rtype: list(AttrDict)
        """
    return PrettyList((f for f in self.frames() if any((re.search(pat, luName) for luName in f.lexUnit))))