import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_edge_list(self):
    G = graphs.StochasticBlockModel(N=100, directed=False)
    v_in, v_out, weights = G.get_edge_list()
    self.assertEqual(G.W[v_in[42], v_out[42]], weights[42])
    G = graphs.StochasticBlockModel(N=100, directed=True)
    self.assertRaises(NotImplementedError, G.get_edge_list)