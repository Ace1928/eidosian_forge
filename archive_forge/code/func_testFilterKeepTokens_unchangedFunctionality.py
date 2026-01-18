from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path
import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts
def testFilterKeepTokens_unchangedFunctionality(self):
    d = Dictionary(self.texts)
    d.filter_extremes(no_below=3, no_above=1.0)
    expected = {'graph', 'trees', 'system', 'user'}
    self.assertEqual(set(d.token2id.keys()), expected)