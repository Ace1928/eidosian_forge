from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_custom_description():
    d = {}

    def record_kwargs(**kwargs):
        d.clear()
        d.update(kwargs)
    c = interactive(record_kwargs, b=widgets.Text(value='text', description='foo'))
    w = c.children[0]
    check_widget(w, cls=widgets.Text, value='text', description='foo')
    w.value = 'different text'
    assert d == {'b': 'different text'}