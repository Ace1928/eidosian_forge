import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def check_example_values(orc_df, expected_df, start=None, stop=None):
    if start is not None or stop is not None:
        expected_df = expected_df[start:stop].reset_index(drop=True)
    assert_frame_equal(orc_df, expected_df, check_dtype=False)