import unittest
import numpy as np
from pygsp import graphs, filters
def test_abspline(self):
    f = filters.Abspline(self._G, Nf=4)
    self._test_methods(f, tight=False)