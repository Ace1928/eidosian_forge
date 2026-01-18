import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_builder_cross_backend_validation(self):
    Store.options(val=self.store_mpl, backend='matplotlib')
    Store.options(val=self.store_bokeh, backend='bokeh')
    Store.set_current_backend('bokeh')
    opts.Curve(line_dash='dotted')
    opts.Curve(linewidth=10)
    err = "In opts.Curve(...), keywords supplied are mixed across backends. Keyword(s) 'linewidth' are invalid for bokeh, 'line_dash' are invalid for matplotlib"
    with pytest.raises(ValueError) as excinfo:
        opts.Curve(linewidth=10, line_dash='dotted')
    assert err in str(excinfo.value)
    err = "In opts.Curve(...), unexpected option 'foobar' for Curve type across all extensions. Similar options for current extension ('bokeh') are: ['toolbar']."
    with pytest.raises(ValueError) as excinfo:
        opts.Curve(foobar=3)
    assert err in str(excinfo.value)
    Store.set_current_backend('matplotlib')
    err = "In opts.Curve(...), unexpected option 'foobar' for Curve type across all extensions. No similar options found."
    with pytest.raises(ValueError) as excinfo:
        opts.Curve(foobar=3)
    assert err in str(excinfo.value)