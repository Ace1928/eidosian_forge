import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
def test_flatten_nested(self):
    nested_list = [[[1, 2, 3], [4, 5]], 6]
    expected = [1, 2, 3, 4, 5, 6]
    self.assertEqual(utils.flatten(nested_list), expected)