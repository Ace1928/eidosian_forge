import unittest
import os
import numpy as np
from skimage import data, img_as_float
from pygsp import graphs, plotting
def test_highlight(self):

    def test(G):
        s = np.arange(G.N)
        G.plot_signal(s, backend='matplotlib', highlight=0)
        G.plot_signal(s, backend='matplotlib', highlight=[0])
        G.plot_signal(s, backend='matplotlib', highlight=[0, 1])
    G = graphs.Ring()
    test(G)
    G = graphs.Ring()
    G.set_coordinates('line1D')
    test(G)
    G = graphs.Torus(Nv=5)
    test(G)