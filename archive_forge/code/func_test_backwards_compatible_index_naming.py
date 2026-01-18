import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_backwards_compatible_index_naming(datadir):
    expected_string = b'carat        cut  color  clarity  depth  table  price     x     y     z\n 0.23      Ideal      E      SI2   61.5   55.0    326  3.95  3.98  2.43\n 0.21    Premium      E      SI1   59.8   61.0    326  3.89  3.84  2.31\n 0.23       Good      E      VS1   56.9   65.0    327  4.05  4.07  2.31\n 0.29    Premium      I      VS2   62.4   58.0    334  4.20  4.23  2.63\n 0.31       Good      J      SI2   63.3   58.0    335  4.34  4.35  2.75\n 0.24  Very Good      J     VVS2   62.8   57.0    336  3.94  3.96  2.48\n 0.24  Very Good      I     VVS1   62.3   57.0    336  3.95  3.98  2.47\n 0.26  Very Good      H      SI1   61.9   55.0    337  4.07  4.11  2.53\n 0.22       Fair      E      VS2   65.1   61.0    337  3.87  3.78  2.49\n 0.23  Very Good      H      VS1   59.4   61.0    338  4.00  4.05  2.39'
    expected = pd.read_csv(io.BytesIO(expected_string), sep='\\s{2,}', index_col=None, header=0, engine='python')
    table = _read_table(datadir / 'v0.7.1.parquet')
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)