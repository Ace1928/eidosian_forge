import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_hash_agg_basic(unary_agg_func_fixture):
    arr1 = pa.array([10.0, 20.0, 30.0, 40.0, 50.0], pa.float64())
    arr2 = pa.array([4, 2, 1, 2, 1], pa.int32())
    arr3 = pa.array([60.0, 70.0, 80.0, 90.0, 100.0], pa.float64())
    arr4 = pa.array([5, 1, 1, 4, 1], pa.int32())
    table1 = pa.table([arr2, arr1], names=['id', 'value'])
    table2 = pa.table([arr4, arr3], names=['id', 'value'])
    table = pa.concat_tables([table1, table2])
    result = table.group_by('id').aggregate([('value', 'mean_udf')])
    expected = table.group_by('id').aggregate([('value', 'mean')]).rename_columns(['id', 'value_mean_udf'])
    assert result.sort_by('id') == expected.sort_by('id')