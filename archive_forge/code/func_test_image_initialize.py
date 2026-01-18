from __future__ import annotations
import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datashader as ds
from datashader.mpl_ext import dsshow
def test_image_initialize():
    plt.figure(dpi=100)
    ax = plt.subplot(111)
    da = dsshow(df, ds.Point('x', 'y'), ax=ax)
    data = da.get_ds_data()
    assert data[0, 0] == 5
    assert data[0, -1] == 5
    assert data[-1, 0] == 5
    assert data[-1, -1] == 5