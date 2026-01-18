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
def test_html_metadata():
    s = '<h1>Test</h1>'
    h = display.HTML(s, metadata={'isolated': True})
    assert h._repr_html_() == (s, {'isolated': True})