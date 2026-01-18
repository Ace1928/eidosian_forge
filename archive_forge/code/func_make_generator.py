import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def make_generator():
    for batch in self.iter_batches(batch_size=batch_size, batch_format='pandas', prefetch_blocks=prefetch_blocks, prefetch_batches=prefetch_batches, drop_last=drop_last, local_shuffle_buffer_size=local_shuffle_buffer_size, local_shuffle_seed=local_shuffle_seed):
        if label_column:
            label_tensor = convert_pandas_to_torch_tensor(batch, [label_column], label_column_dtype, unsqueeze=unsqueeze_label_tensor)
            batch.pop(label_column)
        else:
            label_tensor = None
        if isinstance(feature_columns, dict):
            features_tensor = {key: convert_pandas_to_torch_tensor(batch, feature_columns[key], feature_column_dtypes[key] if isinstance(feature_column_dtypes, dict) else feature_column_dtypes, unsqueeze=unsqueeze_feature_tensors) for key in feature_columns}
        else:
            features_tensor = convert_pandas_to_torch_tensor(batch, columns=feature_columns, column_dtypes=feature_column_dtypes, unsqueeze=unsqueeze_feature_tensors)
        yield (features_tensor, label_tensor)