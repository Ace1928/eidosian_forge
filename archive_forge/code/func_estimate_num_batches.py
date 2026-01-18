import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq
import orjson
from ochat.training_deepspeed.multipack_sampler import MultipackDistributedSampler
def estimate_num_batches(self):
    return self.sampler.estimate_num_batches()