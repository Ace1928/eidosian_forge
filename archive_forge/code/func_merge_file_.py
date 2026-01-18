import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def merge_file_(self, another_file):
    index = MMapIndexedDataset.Index(index_file_path(another_file))
    assert index.dtype == self._dtype
    for size in index.sizes:
        self._sizes.append(size)
    with open(data_file_path(another_file), 'rb') as f:
        shutil.copyfileobj(f, self._data_file)