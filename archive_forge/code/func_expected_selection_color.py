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
def expected_selection_color(element, lnk_sel):
    if lnk_sel.selected_color is not None:
        expected_color = lnk_sel.selected_color
    else:
        expected_color = element.opts.get(group='style')[0].get('color')
    return expected_color