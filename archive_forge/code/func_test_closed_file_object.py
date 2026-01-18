from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
def test_closed_file_object(self):
    file_obj = open(datapath('testcorpus.mm'))
    f = file_obj.closed
    mmcorpus.MmCorpus(file_obj)
    s = file_obj.closed
    self.assertEqual(f, 0)
    self.assertEqual(s, 0)