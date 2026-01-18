import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f'Dataset does not exist: {path}')
        print('Path should be a basename that both .idx and .bin can be appended to get full filenames.')
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f'Unknown dataset implementation: {impl}')
    return None