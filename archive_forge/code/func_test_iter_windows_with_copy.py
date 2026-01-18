import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_iter_windows_with_copy(self):
    texts = [np.array(['this', 'is', 'a'], dtype='object'), np.array(['test', 'document'], dtype='object')]
    windows = list(utils.iter_windows(texts, 2, copy=True))
    windows[0][0] = 'modified'
    self.assertEqual('this', texts[0][0])
    windows[2][0] = 'modified'
    self.assertEqual('test', texts[1][0])