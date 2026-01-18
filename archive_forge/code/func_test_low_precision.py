import logging
import unittest
import numpy as np
from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors
def test_low_precision(self):
    kv = self.load_model(np.float16)
    self.assertAlmostEqual(kv['horse.n.01'][0], -0.00085449)
    self.assertEqual(kv['horse.n.01'][0].dtype, np.float16)