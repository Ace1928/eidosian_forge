import pytest
import modin.pandas as pd
from modin.tests.pandas.utils import default_to_pandas_ignore_string
def dummy_io_method(*args, **kwargs):
    """Dummy method emulating that the code path reached the exchange protocol implementation."""
    raise TestPassed