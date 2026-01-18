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
def test_lines_are_documents(self):
    dirpath = tempfile.mkdtemp()
    lines = ['doc%d text' % i for i in range(5)]
    fpath = os.path.join(dirpath, 'test_file.txt')
    with open(fpath, 'w') as f:
        f.write('\n'.join(lines))
    corpus = textcorpus.TextDirectoryCorpus(dirpath, lines_are_documents=True)
    docs = [doc for doc in corpus.getstream()]
    self.assertEqual(len(lines), corpus.length)
    self.assertEqual(lines, docs)
    corpus.lines_are_documents = False
    docs = [doc for doc in corpus.getstream()]
    self.assertEqual(1, corpus.length)
    self.assertEqual('\n'.join(lines), docs[0])