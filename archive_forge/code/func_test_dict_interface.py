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
def test_dict_interface(self):
    """Test Python 2 dict-like interface in both Python 2 and 3."""
    d = Dictionary(self.texts)
    self.assertTrue(isinstance(d, Mapping))
    self.assertEqual(list(zip(d.keys(), d.values())), list(d.items()))
    self.assertEqual(list(d.items()), list(d.iteritems()))
    self.assertEqual(list(d.keys()), list(d.iterkeys()))
    self.assertEqual(list(d.values()), list(d.itervalues()))