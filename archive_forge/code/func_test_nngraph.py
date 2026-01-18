import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_nngraph(self):
    Xin = np.arange(90).reshape(30, 3)
    dist_types = ['euclidean', 'manhattan', 'max_dist', 'minkowski']
    for dist_type in dist_types:
        if dist_type != 'minkowski':
            graphs.NNGraph(Xin, NNtype='radius', dist_type=dist_type)
            graphs.NNGraph(Xin, NNtype='knn', dist_type=dist_type)
        if dist_type != 'max_dist':
            graphs.NNGraph(Xin, use_flann=True, NNtype='knn', dist_type=dist_type)