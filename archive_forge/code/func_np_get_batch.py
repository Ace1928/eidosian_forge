import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def np_get_batch(indices, dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, return_dict=False):
    if not isinstance(indices, np.ndarray):
        indices = indices.numpy()
    is_batched = True
    if isinstance(indices, np.integer):
        batch = dataset[indices.item()]
        is_batched = False
    elif np.all(np.diff(indices) == 1):
        batch = dataset[indices[0]:indices[-1] + 1]
    elif isinstance(indices, np.ndarray):
        batch = dataset[indices]
    else:
        raise RuntimeError('Unexpected type for indices: {}'.format(type(indices)))
    if cols_to_retain is not None:
        batch = {key: value for key, value in batch.items() if key in cols_to_retain or key in ('label', 'label_ids', 'labels')}
    if is_batched:
        actual_size = len(list(batch.values())[0])
        batch = [{key: value[i] for key, value in batch.items()} for i in range(actual_size)]
    batch = collate_fn(batch, **collate_fn_args)
    if return_dict:
        out_batch = {}
        for col, cast_dtype in columns_to_np_types.items():
            array = np.array(batch[col])
            array = array.astype(cast_dtype)
            out_batch[col] = array
    else:
        out_batch = []
        for col, cast_dtype in columns_to_np_types.items():
            array = np.array(batch[col])
            array = array.astype(cast_dtype)
            out_batch.append(array)
    return out_batch