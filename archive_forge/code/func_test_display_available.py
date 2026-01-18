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
def test_display_available():
    """
    Test that display is available without import

    We don't really care if it's in builtin or anything else, but it should
    always be available.
    """
    ip = get_ipython()
    with AssertNotPrints('NameError'):
        ip.run_cell('display')
    try:
        ip.run_cell('del display')
    except NameError:
        pass
    with AssertNotPrints('NameError'):
        ip.run_cell('display')