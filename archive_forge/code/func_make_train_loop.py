import sys
import time
from typing import Optional
import numpy as np
import ray
from ray import train
from ray.air.config import DatasetConfig, ScalingConfig
from ray.data import Dataset, DataIterator, Preprocessor
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import DataConfig
from ray.util.annotations import Deprecated, DeveloperAPI
@staticmethod
def make_train_loop(num_epochs: int, prefetch_batches: int, prefetch_blocks: int, batch_size: Optional[int]):
    """Make a debug train loop that runs for the given amount of epochs."""

    def train_loop_per_worker():
        import pandas as pd
        rank = train.get_context().get_world_rank()
        data_shard = train.get_dataset_shard('train')
        start = time.perf_counter()
        epochs_read, batches_read, bytes_read = (0, 0, 0)
        batch_delays = []
        print('Starting train loop on worker', rank)
        for epoch in range(num_epochs):
            epochs_read += 1
            batch_start = time.perf_counter()
            for batch in data_shard.iter_batches(prefetch_batches=prefetch_batches, prefetch_blocks=prefetch_blocks, batch_size=batch_size):
                batch_delay = time.perf_counter() - batch_start
                batch_delays.append(batch_delay)
                batches_read += 1
                if isinstance(batch, pd.DataFrame):
                    bytes_read += int(batch.memory_usage(index=True, deep=True).sum())
                elif isinstance(batch, np.ndarray):
                    bytes_read += batch.nbytes
                elif isinstance(batch, dict):
                    for arr in batch.values():
                        bytes_read += arr.nbytes
                else:
                    bytes_read += sys.getsizeof(batch)
                train.report(dict(bytes_read=bytes_read, batches_read=batches_read, epochs_read=epochs_read, batch_delay=batch_delay))
                batch_start = time.perf_counter()
        delta = time.perf_counter() - start
        print('Time to read all data', delta, 'seconds')
        print('P50/P95/Max batch delay (s)', np.quantile(batch_delays, 0.5), np.quantile(batch_delays, 0.95), np.max(batch_delays))
        print('Num epochs read', epochs_read)
        print('Num batches read', batches_read)
        print('Num bytes read', round(bytes_read / (1024 * 1024), 2), 'MiB')
        print('Mean throughput', round(bytes_read / (1024 * 1024) / delta, 2), 'MiB/s')
        if rank == 0:
            print('Ingest stats from rank=0:\n\n{}'.format(data_shard.stats()))
    return train_loop_per_worker