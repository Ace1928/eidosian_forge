import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_edge_paths(self):
    edge_test1 = self.build_operands('eb,cb,fb->cef')
    path, path_str = np.einsum_path(*edge_test1, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])
    path, path_str = np.einsum_path(*edge_test1, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])
    edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
    path, path_str = np.einsum_path(*edge_test2, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])
    path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])
    edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
    path, path_str = np.einsum_path(*edge_test3, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
    path, path_str = np.einsum_path(*edge_test3, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
    edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
    path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])
    path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])
    edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->', size_dict={'a': 20, 'b': 20, 'c': 20, 'd': 20})
    path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
    self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])
    path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
    self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])