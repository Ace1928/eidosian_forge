import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_filters_equivalency(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    integer_keys = [0, 1]
    string_keys = ['a', 'b', 'c']
    boolean_keys = [True, False]
    partition_spec = [['integer', integer_keys], ['string', string_keys], ['boolean', boolean_keys]]
    df = pd.DataFrame({'integer': np.array(integer_keys, dtype='i4').repeat(15), 'string': np.tile(np.tile(np.array(string_keys, dtype=object), 5), 2), 'boolean': np.tile(np.tile(np.array(boolean_keys, dtype='bool'), 5), 3)}, columns=['integer', 'string', 'boolean'])
    _generate_partition_directories(fs, base_path, partition_spec, df)
    dataset = pq.ParquetDataset(base_path, filesystem=fs, filters=[('integer', '=', 1), ('string', '!=', 'b'), ('boolean', '==', 'True')])
    table = dataset.read()
    result_df = table.to_pandas().reset_index(drop=True)
    assert 0 not in result_df['integer'].values
    assert 'b' not in result_df['string'].values
    assert False not in result_df['boolean'].values
    filters = [[('integer', '=', 1), ('string', '!=', 'b'), ('boolean', '==', 'True')], [('integer', '=', 0), ('boolean', '==', 'False')]]
    dataset = pq.ParquetDataset(base_path, filesystem=fs, filters=filters)
    table = dataset.read()
    result_df = table.to_pandas().reset_index(drop=True)
    df_filter_1 = (result_df['integer'] == 1) & (result_df['string'] != 'b') & (result_df['boolean'] == 'True')
    df_filter_2 = (np.array(result_df['integer']) == 0) & (result_df['boolean'] == 'False')
    assert df_filter_1.sum() > 0
    assert df_filter_2.sum() > 0
    assert result_df.shape[0] == df_filter_1.sum() + df_filter_2.sum()
    for filters in [[[('string', '==', b'1\x00a')]], [[('string', '==', '1\x00a')]]]:
        dataset = pq.ParquetDataset(base_path, filesystem=fs, filters=filters)
        assert dataset.read().num_rows == 0