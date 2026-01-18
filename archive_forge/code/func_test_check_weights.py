import unittest
import numpy as np
from scipy import sparse
from pygsp import graphs, utils
def test_check_weights(G, w_c):
    self.assertEqual(G.check_weights(), w_c)