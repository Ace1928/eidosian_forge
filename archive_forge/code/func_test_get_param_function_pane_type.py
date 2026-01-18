import asyncio
import os
import pandas as pd
import param
import pytest
from bokeh.models import (
from packaging.version import Version
from panel import config
from panel.depends import bind
from panel.io.state import set_curdoc, state
from panel.layout import Row, Tabs
from panel.models import HTML as BkHTML
from panel.pane import (
from panel.param import (
from panel.tests.util import mpl_available, mpl_figure
from panel.widgets import (
def test_get_param_function_pane_type():
    test = View()

    def view(a):
        return Div(text='%d' % a)
    assert PaneBase.get_pane_type(view) is not ParamFunction
    assert PaneBase.get_pane_type(param.depends(test.param.a)(view)) is ParamFunction