import os
import unittest
import random
import shutil
import numpy as np
from scipy import sparse
from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus
def test_getitem_dense2gensim(self):
    corpus = ShardedCorpus(self.tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=False, gensim=True)
    item = corpus[3]
    self.assertTrue(isinstance(item, list))
    self.assertTrue(isinstance(item[0], tuple))
    dslice = corpus[2:6]
    self.assertTrue(next(dslice) == corpus[2])
    dslice = list(dslice)
    self.assertTrue(isinstance(dslice, list))
    self.assertTrue(isinstance(dslice[0], list))
    self.assertTrue(isinstance(dslice[0][0], tuple))
    iscorp, _ = is_corpus(dslice)
    self.assertTrue(iscorp, 'Is the object returned by slice notation a gensim corpus?')
    ilist = corpus[[2, 3, 4, 5]]
    self.assertTrue(next(ilist) == corpus[2])
    ilist = list(ilist)
    self.assertTrue(isinstance(ilist, list))
    self.assertTrue(isinstance(ilist[0], list))
    self.assertTrue(isinstance(ilist[0][0], tuple))
    self.assertEqual(len(ilist), len(dslice))
    for i in range(len(ilist)):
        self.assertEqual(len(ilist[i]), len(dslice[i]), 'Row %d: dims %d/%d' % (i, len(ilist[i]), len(dslice[i])))
        for j in range(len(ilist[i])):
            self.assertEqual(ilist[i][j], dslice[i][j], 'ilist[%d][%d] = %s ,dslice[%d][%d] = %s' % (i, j, str(ilist[i][j]), i, j, str(dslice[i][j])))
    iscorp, _ = is_corpus(ilist)
    self.assertTrue(iscorp, 'Is the object returned by list notation a gensim corpus?')