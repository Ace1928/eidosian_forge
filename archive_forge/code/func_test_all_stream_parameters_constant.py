from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
def test_all_stream_parameters_constant():
    all_stream_cls = [v for v in globals().values() if isinstance(v, type) and issubclass(v, Stream)]
    for stream_cls in all_stream_cls:
        for name, p in stream_cls.param.objects().items():
            if name == 'name':
                continue
            if p.constant != True:
                raise TypeError(f'Parameter {name} of stream {stream_cls.__name__} not declared constant')