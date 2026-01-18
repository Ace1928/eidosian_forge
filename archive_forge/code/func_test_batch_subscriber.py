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
def test_batch_subscriber(self):
    subscriber = _TestSubscriber()
    positionX = PointerX(subscribers=[subscriber])
    positionY = PointerY(subscribers=[subscriber])
    positionX.update(x=5)
    positionY.update(y=10)
    Stream.trigger([positionX, positionY])
    self.assertEqual(subscriber.kwargs, dict(x=5, y=10))
    self.assertEqual(subscriber.call_count, 1)