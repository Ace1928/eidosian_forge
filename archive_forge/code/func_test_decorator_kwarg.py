from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_decorator_kwarg(clear_display):
    with patch.object(interaction, 'display', record_display):

        @interact(a=5)
        def foo(a):
            pass
    assert len(displayed) == 1
    w = displayed[0].children[0]
    check_widget(w, cls=widgets.IntSlider, value=5)