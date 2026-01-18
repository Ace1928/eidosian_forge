import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
@staticmethod
def distribute_batches(dataset, batch_size, drop_remainder, num_workers, shuffle):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    num_samples = len(indices)
    incomplete_batch_cutoff = num_samples - num_samples % batch_size
    indices, last_incomplete_batch = np.split(indices, [incomplete_batch_cutoff])
    if drop_remainder or len(last_incomplete_batch) == 0:
        last_incomplete_batch = None
    indices = indices.reshape(-1, batch_size)
    num_batches = len(indices)
    final_batches_cutoff = num_batches - num_batches % num_workers
    indices, final_batches = np.split(indices, [final_batches_cutoff])
    indices = indices.reshape(-1, num_workers, batch_size)
    per_worker_indices = np.split(indices, indices.shape[1], axis=1)
    per_worker_indices = [np.squeeze(worker_indices, 1) for worker_indices in per_worker_indices]
    for i in range(len(final_batches)):
        per_worker_indices[i] = np.concatenate([per_worker_indices[i], final_batches[i].reshape(1, -1)], axis=0)
    if last_incomplete_batch is not None:
        incomplete_batch_worker_idx = len(final_batches)
    else:
        incomplete_batch_worker_idx = None
    return (per_worker_indices, last_incomplete_batch, incomplete_batch_worker_idx)