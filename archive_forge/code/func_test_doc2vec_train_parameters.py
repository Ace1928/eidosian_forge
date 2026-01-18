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
def test_doc2vec_train_parameters(self):
    model = doc2vec.Doc2Vec(vector_size=50)
    model.build_vocab(corpus_iterable=list_corpus)
    self.assertRaises(TypeError, model.train, corpus_file=11111)
    self.assertRaises(TypeError, model.train, corpus_iterable=11111)
    self.assertRaises(TypeError, model.train, corpus_iterable=sentences, corpus_file='test')
    self.assertRaises(TypeError, model.train, corpus_iterable=None, corpus_file=None)
    self.assertRaises(TypeError, model.train, corpus_file=sentences)