from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_multi_thread_path_multipart_read_csv(all_parsers):
    num_tasks = 4
    num_rows = 48
    parser = all_parsers
    file_name = '__thread_pool_reader__.csv'
    df = DataFrame({'a': np.random.default_rng(2).random(num_rows), 'b': np.random.default_rng(2).random(num_rows), 'c': np.random.default_rng(2).random(num_rows), 'd': np.random.default_rng(2).random(num_rows), 'e': np.random.default_rng(2).random(num_rows), 'foo': ['foo'] * num_rows, 'bar': ['bar'] * num_rows, 'baz': ['baz'] * num_rows, 'date': pd.date_range('20000101 09:00:00', periods=num_rows, freq='s'), 'int': np.arange(num_rows, dtype='int64')})
    with tm.ensure_clean(file_name) as path:
        df.to_csv(path)
        final_dataframe = _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks)
        tm.assert_frame_equal(df, final_dataframe)