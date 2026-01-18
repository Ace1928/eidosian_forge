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
def test_filename_filtering(self):
    dirpath = self.write_one_level('test1.log', 'test1.txt', 'test2.log', 'other1.log')
    corpus = textcorpus.TextDirectoryCorpus(dirpath, pattern='test.*\\.log')
    filenames = list(corpus.iter_filepaths())
    expected = [os.path.join(dirpath, name) for name in ('test1.log', 'test2.log')]
    self.assertEqual(sorted(expected), sorted(filenames))
    corpus.pattern = '.*.txt'
    filenames = list(corpus.iter_filepaths())
    expected = [os.path.join(dirpath, 'test1.txt')]
    self.assertEqual(expected, filenames)
    corpus.pattern = None
    corpus.exclude_pattern = '.*.log'
    filenames = list(corpus.iter_filepaths())
    self.assertEqual(expected, filenames)