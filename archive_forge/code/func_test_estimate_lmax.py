import unittest
import numpy as np
from scipy import sparse
from pygsp import graphs, utils
def test_estimate_lmax(G, lmax):
    G.estimate_lmax()
    self.assertTrue(lmax <= G.lmax and G.lmax <= 1.02 * lmax)