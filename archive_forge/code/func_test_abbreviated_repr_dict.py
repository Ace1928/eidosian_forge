import sys
from collections import OrderedDict
import param
from bokeh.models import Div
from panel.depends import bind
from panel.io.notebook import render_mimebundle
from panel.pane import PaneBase
from panel.tests.util import mpl_available
from panel.util import (
def test_abbreviated_repr_dict():
    assert abbreviated_repr({'key': 'some really, really long string'}) == "{'key': 'some really, ...}"