from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
def test_int_doctags(self):
    """Test doc2vec doctag alternatives"""
    corpus = DocsLeeCorpus()
    model = doc2vec.Doc2Vec(min_count=1)
    model.build_vocab(corpus)
    self.assertEqual(len(model.dv.vectors), 300)
    self.assertEqual(model.dv[0].shape, (100,))
    self.assertEqual(model.dv[np.int64(0)].shape, (100,))
    self.assertRaises(KeyError, model.__getitem__, '_*0')