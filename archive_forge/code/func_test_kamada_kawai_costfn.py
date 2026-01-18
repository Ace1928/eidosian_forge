import pytest
import networkx as nx
def test_kamada_kawai_costfn(self):
    invdist = 1 / np.array([[0.1, 2.1, 1.7], [2.1, 0.2, 0.6], [1.7, 0.6, 0.3]])
    meanwt = 0.3
    pos = np.array([[1.3, -3.2], [2.7, -0.3], [5.1, 2.5]])
    self.check_kamada_kawai_costfn(pos, invdist, meanwt, 2)
    pos = np.array([[0.9, 8.6, -8.7], [-10, -0.5, -7.1], [9.1, -8.1, 1.6]])
    self.check_kamada_kawai_costfn(pos, invdist, meanwt, 3)