from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
def test_diagonal_kwargs_mutually_exclusive(self):
    with self.assertRaises(TypeError):
        scatter_matrix(self.df, diagonal_kwds=dict(a=1), hist_kwds=dict(a=1))
    with self.assertRaises(TypeError):
        scatter_matrix(self.df, diagonal_kwds=dict(a=1), density_kwds=dict(a=1))
    with self.assertRaises(TypeError):
        scatter_matrix(self.df, density_kwds=dict(a=1), hist_kwds=dict(a=1))