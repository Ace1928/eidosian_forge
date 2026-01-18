import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f'Dataset does not exist: {path}')
        print('Path should be a basename that both .idx and .bin can be appended to get full filenames.')
        return None