from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_iterable_tuple():
    values = [(3, 300), (1, 100), (2, 200)]
    first = values[0][1]
    c = interactive(f, lis=iter(values))
    assert len(c.children) == 2
    d = dict(cls=widgets.Dropdown, value=first, options=tuple(values), _options_labels=('3', '1', '2'), _options_values=(300, 100, 200))
    check_widget_children(c, lis=d)