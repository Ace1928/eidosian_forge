import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
def is_panel_layout(attr):
    layout = getattr(pn.layout, attr)
    return isclass(layout) and issubclass(layout, pn.layout.Panel)