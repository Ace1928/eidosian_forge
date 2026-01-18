from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
def test_diagonal_kwargs(self):
    sm = scatter_matrix(self.df, diagonal_kwds=dict(line_color='red'))
    self.assertEqual(sm['a', 'a'].opts.get().kwargs['line_color'], 'red')