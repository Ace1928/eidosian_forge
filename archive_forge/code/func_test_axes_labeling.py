from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
@pytest.mark.matplotlib
def test_axes_labeling(close_figures):
    from numpy.random import rand
    key_set = (['male', 'female'], ['old', 'adult', 'young'], ['worker', 'unemployed'], ['yes', 'no'])
    keys = list(product(*key_set))
    data = dict(zip(keys, rand(len(keys))))
    lab = lambda k: ''.join((s[0] for s in k))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    mosaic(data, ax=ax1, labelizer=lab, horizontal=True, label_rotation=45)
    mosaic(data, ax=ax2, labelizer=lab, horizontal=False, label_rotation=[0, 45, 90, 0])
    fig.suptitle('correct alignment of the axes labels')