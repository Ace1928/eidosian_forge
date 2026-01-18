from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
@staticmethod
def get_value_with_key_type(d, hvtype):
    for k, v in d.items():
        if isinstance(k, hvtype) or (isinstance(k, hv.DynamicMap) and k.type == hvtype):
            return v
    raise KeyError(f'No key with type {hvtype}')