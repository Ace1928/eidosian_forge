import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_displayobject_repr():
    h = display.HTML('<br />')
    assert repr(h) == '<IPython.core.display.HTML object>'
    h._show_mem_addr = True
    assert repr(h) == object.__repr__(h)
    h._show_mem_addr = False
    assert repr(h) == '<IPython.core.display.HTML object>'
    j = display.Javascript('')
    assert repr(j) == '<IPython.core.display.Javascript object>'
    j._show_mem_addr = True
    assert repr(j) == object.__repr__(j)
    j._show_mem_addr = False
    assert repr(j) == '<IPython.core.display.Javascript object>'