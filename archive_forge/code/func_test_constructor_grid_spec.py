import json
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from panel.layout import GridSpec, Row
from panel.pane import HoloViews, Markdown
from panel.template import (
from panel.template.base import BasicTemplate
from panel.widgets import FloatSlider
from .util import hv_available
def test_constructor_grid_spec():
    item = Markdown('Hello World')
    grid = GridSpec(ncols=12)
    grid[0:2, 3:4] = item
    ReactTemplate(main=grid)