import sys
from unittest import SkipTest
from parameterized import parameterized
import colorcet as cc
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
from holoviews import Store, render
from holoviews.element import Image, QuadMesh, Points
from holoviews.core.spaces import DynamicMap
from holoviews.core.overlay import Overlay
from holoviews.element.chart import Scatter
from holoviews.element.comparison import ComparisonTestCase
from hvplot.converter import HoloViewsConverter
from hvplot.tests.util import makeTimeDataFrame
from packaging.version import Version
@parameterized.expand([('points', Points), ('scatter', Scatter)])
def test_downsample_resample_when(self, kind, eltype):
    df = pd.DataFrame(np.random.multivariate_normal((0, 0), [[0.1, 0.1], [0.1, 1.0]], (5000,))).rename({0: 'x', 1: 'y'}, axis=1)
    dmap = df.hvplot(kind=kind, x='x', y='y', resample_when=1000, downsample=True)
    assert isinstance(dmap, DynamicMap)
    render(dmap)
    overlay = dmap.items()[0][1]
    assert isinstance(overlay, Overlay)
    downsampled = overlay.get(0)
    assert isinstance(downsampled, eltype)
    assert len(downsampled) > 0
    element = overlay.get(1)
    assert isinstance(element, eltype)
    assert len(element) == 0