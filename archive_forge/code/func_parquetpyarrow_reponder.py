from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def parquetpyarrow_reponder(df):
    return df.to_parquet(index=False, engine='pyarrow')