from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_call_interact_kwargs(clear_display):

    def foo(a='default'):
        pass
    with patch.object(interaction, 'display', record_display):
        ifoo = interact(foo, a=10)
    assert len(displayed) == 1
    w = displayed[0].children[0]
    check_widget(w, cls=widgets.IntSlider, value=10)