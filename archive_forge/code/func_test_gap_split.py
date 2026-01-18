from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
def test_gap_split():
    pure_square = [0.0, 0.0, 1.0, 1.0]
    conf_h = dict(proportion=[1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), pure_square)
    h_2split = [(0.0, 0.0, 0.25, 1.0), (0.75, 0.0, 0.25, 1.0)]
    conf_h = dict(proportion=[1, 1], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)
    h_2split = [(0.0, 0.0, 1 / 6, 1.0), (0.5 + 1 / 6, 0.0, 1 / 3, 1.0)]
    conf_h = dict(proportion=[1, 2], gap=1.0, horizontal=True)
    eq(_split_rect(*pure_square, **conf_h), h_2split)