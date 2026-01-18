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
def test_jsoninit_class_from_env_var():
    os.environ['PARAM_JSON_INIT'] = '{"a": 1}'
    json_init = JSONInit()

    class Test(param.Parameterized):
        a = param.Integer()
    json_init(Test)
    assert Test.a == 1
    del os.environ['PARAM_JSON_INIT']