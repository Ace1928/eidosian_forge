import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
def test_datetime_timezone_tzinfo():
    value = datetime.datetime(2018, 1, 1, 1, 23, 45, tzinfo=datetime.timezone.utc)
    df = pd.DataFrame({'foo': [value]})
    _roundtrip_pandas_dataframe(df, write_kwargs={})