import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
@pytest.fixture
def restore_backend():
    """Restore the plotting backend to matplotlib"""
    with pandas.option_context('plotting.backend', 'matplotlib'):
        yield