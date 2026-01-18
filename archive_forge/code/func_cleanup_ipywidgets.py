import pytest
import traitlets
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from panel.layout import Row
from panel.pane.ipywidget import Reacton
from panel.tests.util import serve_component, wait_until
@pytest.fixture(scope='module', autouse=True)
def cleanup_ipywidgets():
    old_models = dict(Model.model_class_reverse_map)
    yield
    _default_resolver._known_models = old_models