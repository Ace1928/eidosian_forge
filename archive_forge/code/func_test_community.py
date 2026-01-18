import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_community(self):
    graphs.Community()
    graphs.Community(comm_density=0.2)
    graphs.Community(k_neigh=5)
    graphs.Community(N=100, Nc=3, comm_sizes=[20, 50, 30])