from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_interact_call():
    w = interact.widget(f)
    w.update()
    w = interact_manual.widget(f)
    w.update()