import unittest
import numpy as np
from pygsp import graphs, filters
def test_halfcosine(self):
    f = filters.HalfCosine(self._G, Nf=4)
    self._test_methods(f, tight=True)