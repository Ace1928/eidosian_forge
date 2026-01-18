import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_eigendecompositions(self):
    G = graphs.Logo()
    U1, e1, V1 = scipy.linalg.svd(G.L.toarray())
    U2, e2, V2 = np.linalg.svd(G.L.toarray())
    e3, U3 = np.linalg.eig(G.L.toarray())
    e4, U4 = scipy.linalg.eig(G.L.toarray())
    e5, U5 = np.linalg.eigh(G.L.toarray())
    e6, U6 = scipy.linalg.eigh(G.L.toarray())

    def correct_sign(U):
        signs = np.sign(U[0, :])
        signs[signs == 0] = 1
        return U * signs
    U1 = correct_sign(U1)
    U2 = correct_sign(U2)
    U3 = correct_sign(U3)
    U4 = correct_sign(U4)
    U5 = correct_sign(U5)
    U6 = correct_sign(U6)
    V1 = correct_sign(V1.T)
    V2 = correct_sign(V2.T)
    inds3 = np.argsort(e3)[::-1]
    inds4 = np.argsort(e4)[::-1]
    np.testing.assert_allclose(e2, e1)
    np.testing.assert_allclose(e3[inds3], e1, atol=1e-12)
    np.testing.assert_allclose(e4[inds4], e1, atol=1e-12)
    np.testing.assert_allclose(e5[::-1], e1, atol=1e-12)
    np.testing.assert_allclose(e6[::-1], e1, atol=1e-12)
    np.testing.assert_allclose(U2, U1, atol=1e-12)
    np.testing.assert_allclose(V1, U1, atol=1e-12)
    np.testing.assert_allclose(V2, U1, atol=1e-12)
    np.testing.assert_allclose(U3[:, inds3], U1, atol=1e-10)
    np.testing.assert_allclose(U4[:, inds4], U1, atol=1e-10)
    np.testing.assert_allclose(U5[:, ::-1], U1, atol=1e-10)
    np.testing.assert_allclose(U6[:, ::-1], U1, atol=1e-10)