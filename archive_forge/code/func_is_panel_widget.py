import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
def is_panel_widget(attr):
    widget = getattr(pn.widgets, attr)
    return isclass(widget) and issubclass(widget, pn.widgets.Widget)