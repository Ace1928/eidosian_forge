from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_int_range_logic():
    irsw = widgets.IntRangeSlider
    w = irsw(value=(2, 4), min=0, max=6)
    check_widget(w, cls=irsw, value=(2, 4), min=0, max=6)
    w.upper = 3
    w.max = 3
    check_widget(w, cls=irsw, value=(2, 3), min=0, max=3)
    w.min = 0
    w.max = 6
    w.lower = 2
    w.upper = 4
    check_widget(w, cls=irsw, value=(2, 4), min=0, max=6)
    w.value = (0, 1)
    check_widget(w, cls=irsw, value=(0, 1), min=0, max=6)
    w.value = (5, 6)
    check_widget(w, cls=irsw, value=(5, 6), min=0, max=6)
    w.lower = 2
    check_widget(w, cls=irsw, value=(2, 6), min=0, max=6)
    with pytest.raises(TraitError):
        w.min = 7
    with pytest.raises(TraitError):
        w.max = -1
    w = irsw(min=2, max=3, value=(2, 3))
    check_widget(w, min=2, max=3, value=(2, 3))
    w = irsw(min=100, max=200, value=(125, 175))
    check_widget(w, value=(125, 175))
    with pytest.raises(TraitError):
        irsw(min=2, max=1)