from unittest.mock import patch
import os
from collections import OrderedDict
import pytest
import ipywidgets as widgets
from traitlets import TraitError, Float
from ipywidgets import (interact, interact_manual, interactive,
from .utils import setup, teardown
def test_interact_manual_nocall():
    callcount = 0

    def calltest(testarg):
        callcount += 1
    c = interact.options(manual=True)(calltest, testarg=5).widget
    c.children[0].value = 10
    assert callcount == 0