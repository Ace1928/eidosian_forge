from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_multi_thread_string_io_read_csv(all_parsers):
    parser = all_parsers
    max_row_range = 100
    num_files = 10
    bytes_to_df = ('\n'.join([f'{i:d},{i:d},{i:d}' for i in range(max_row_range)]).encode() for _ in range(num_files))
    with ExitStack() as stack:
        files = [stack.enter_context(BytesIO(b)) for b in bytes_to_df]
        pool = stack.enter_context(ThreadPool(8))
        results = pool.map(parser.read_csv, files)
        first_result = results[0]
        for result in results:
            tm.assert_frame_equal(first_result, result)