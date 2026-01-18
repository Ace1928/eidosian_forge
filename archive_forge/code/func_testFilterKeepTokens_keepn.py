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
def testFilterKeepTokens_keepn(self):
    d = Dictionary(self.texts)
    d.add_documents([['worda'], ['wordb']])
    d.filter_extremes(keep_n=5, no_below=0, no_above=1.0, keep_tokens=['worda'])
    expected = {'graph', 'trees', 'system', 'user', 'worda'}
    self.assertEqual(set(d.token2id.keys()), expected)