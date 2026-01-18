from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def check_widget_children(container, **to_check):
    """Check that widgets are created as expected"""
    widgets = {}
    for w in container.children:
        if not isinstance(w, Output):
            widgets[w.description] = w
    for key, d in to_check.items():
        assert key in widgets
        check_widget(widgets[key], **d)