import unittest
import numpy as np
from pygsp import graphs, filters
def test_itersine(self):
    f = filters.Itersine(self._G, Nf=4)
    self._test_methods(f, tight=True)