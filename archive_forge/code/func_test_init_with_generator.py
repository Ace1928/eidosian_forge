import os
import unittest
import random
import shutil
import numpy as np
from scipy import sparse
from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus
def test_init_with_generator(self):

    def data_generator():
        yield [(0, 1)]
        yield [(1, 1)]
    gen_tmp_fname = self.tmp_fname + '.generator'
    corpus = ShardedCorpus(gen_tmp_fname, data_generator(), dim=2)
    self.assertEqual(2, len(corpus))
    self.assertEqual(1, corpus[0][0])