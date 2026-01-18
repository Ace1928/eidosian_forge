import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def send_batch_to_parent(indices):
    batch = np_get_batch(indices=indices, dataset=dataset, cols_to_retain=cols_to_retain, collate_fn=collate_fn, collate_fn_args=collate_fn_args, columns_to_np_types=columns_to_np_types, return_dict=True)
    out_arrays = {}
    with SharedMemoryContext() as batch_shm_ctx:
        for col, cast_dtype in columns_to_np_types.items():
            array = batch[col]
            if col in string_columns:
                array = array.view('U1').reshape(array.shape + (-1,))
            shape_arrays[col][:] = array.shape
            out_arrays[col] = batch_shm_ctx.get_array(f'{worker_name}_{col}', shape=array.shape, dtype=cast_dtype, create=True)
            out_arrays[col][:] = array
        array_ready_event.set()
        array_loaded_event.wait()
        array_loaded_event.clear()