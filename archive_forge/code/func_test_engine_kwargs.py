import contextlib
import pytest
from pandas.compat import is_platform_windows
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.excel import ExcelWriter
@pytest.mark.parametrize('nan_inf_to_errors', [True, False])
def test_engine_kwargs(ext, nan_inf_to_errors):
    engine_kwargs = {'options': {'nan_inf_to_errors': nan_inf_to_errors}}
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine='xlsxwriter', engine_kwargs=engine_kwargs) as writer:
            assert writer.book.nan_inf_to_errors == nan_inf_to_errors