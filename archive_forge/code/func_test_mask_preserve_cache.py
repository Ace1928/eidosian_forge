import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data
@pytest.mark.parametrize('row_labels', [[0, 2], slice(None)])
@pytest.mark.parametrize('col_labels', [[0, 2], slice(None)])
@pytest.mark.parametrize('is_length_future', [False, True])
@pytest.mark.parametrize('is_width_future', [False, True])
def test_mask_preserve_cache(row_labels, col_labels, is_length_future, is_width_future):

    def deserialize(obj):
        if is_future(obj):
            return get_func(obj)
        return obj

    def compute_length(indices, length):
        if not isinstance(indices, slice):
            return len(indices)
        return compute_sliced_len(indices, length)
    df = pandas.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [9, 10, 11, 12]})
    obj_id = put_func(df)
    partition_shape = [put_func(len(df)) if is_length_future else len(df), put_func(len(df.columns)) if is_width_future else len(df.columns)]
    source_partition = PartitionClass(obj_id, *partition_shape)
    masked_partition = source_partition.mask(row_labels=row_labels, col_labels=col_labels)
    expected_length = compute_length(row_labels, len(df))
    expected_width = compute_length(col_labels, len(df.columns))
    assert expected_length == deserialize(masked_partition._length_cache)
    assert expected_width == deserialize(masked_partition._width_cache)
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()
    expected_length, expected_width = [masked_partition._length_cache, masked_partition._width_cache]
    masked_partition._length_cache = None
    masked_partition._width_cache = None
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()