import pytest
import modin.pandas as pd
from modin.tests.pandas.utils import default_to_pandas_ignore_string
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_basic_io(get_unique_base_execution):
    """Test that the protocol IO functions actually reach their implementation with no errors."""

    class TestPassed(BaseException):
        pass

    def dummy_io_method(*args, **kwargs):
        """Dummy method emulating that the code path reached the exchange protocol implementation."""
        raise TestPassed
    query_compiler_cls = get_unique_base_execution
    query_compiler_cls.from_dataframe = dummy_io_method
    query_compiler_cls.to_dataframe = dummy_io_method
    from modin.pandas.io import from_dataframe
    with pytest.raises(TestPassed):
        from_dataframe(None)
    with pytest.raises(TestPassed):
        pd.DataFrame([[1]]).__dataframe__()