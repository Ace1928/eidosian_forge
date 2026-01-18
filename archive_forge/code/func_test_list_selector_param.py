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
def test_list_selector_param(document, comm):

    class Test(param.Parameterized):
        a = param.ListSelector(default=['b', 1], objects=[1, 'b', 'c'])
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    slider = model.children[1]
    assert isinstance(slider, MultiSelect)
    assert slider.options == ['1', 'b', 'c']
    assert slider.value == ['b', '1']
    assert slider.disabled == False
    test.a = ['c', 1]
    assert slider.value == ['c', '1']
    a_param = test.param['a']
    a_param.objects = ['c', 'd', 1]
    assert slider.options == ['c', 'd', '1']
    a_param.constant = True
    assert slider.disabled == True
    test_pane._cleanup(model)
    a_param.constant = False
    a_param.objects = [1, 'c', 'd']
    test.a = ['d']
    assert slider.value == ['c', '1']
    assert slider.options == ['c', 'd', '1']
    assert slider.disabled == True