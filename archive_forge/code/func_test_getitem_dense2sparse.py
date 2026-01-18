import os
import unittest
import random
import shutil
import numpy as np
from scipy import sparse
from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus
def test_getitem_dense2sparse(self):
    corpus = ShardedCorpus(self.tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=False, sparse_retrieval=True)
    item = corpus[3]
    self.assertTrue(isinstance(item, sparse.csr_matrix))
    self.assertEqual(item.shape, (1, corpus.dim))
    dslice = corpus[2:6]
    self.assertTrue(isinstance(dslice, sparse.csr_matrix))
    self.assertEqual(dslice.shape, (4, corpus.dim))
    ilist = corpus[[2, 3, 4, 5]]
    self.assertTrue(isinstance(ilist, sparse.csr_matrix))
    self.assertEqual(ilist.shape, (4, corpus.dim))
    self.assertEqual((ilist != dslice).getnnz(), 0)