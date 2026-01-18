from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def compute_impl(self, X: np.ndarray, row_num: int, row_size: int, frequencies: List[int], max_gram_length=None, max_skip_count=None, min_gram_length=None, mode=None, ngram_counts=None, ngram_indexes=None, pool_int64s=None, pool_strings=None, weights=None) -> None:
    if len(X.shape) > 1:
        X_flat = X[row_num]
    else:
        X_flat = X
    row_begin = 0
    row_end = row_begin + row_size
    max_skip_distance = max_skip_count + 1
    start_ngram_size = min_gram_length
    for skip_distance in range(1, max_skip_distance + 1):
        ngram_start = row_begin
        ngram_row_end = row_end
        while ngram_start < ngram_row_end:
            at_least_this = ngram_start + skip_distance * (start_ngram_size - 1)
            if at_least_this >= ngram_row_end:
                break
            ngram_item = ngram_start
            int_map = self.int64_map_
            ngram_size = 1
            while int_map.has_leaves() and ngram_size <= max_gram_length and (ngram_item < ngram_row_end):
                val = X_flat[ngram_item]
                hit = int_map.find(val)
                if hit is None:
                    break
                hit = int_map[val].id_
                if ngram_size >= start_ngram_size and hit != 0:
                    self.increment_count(hit, row_num, frequencies)
                int_map = int_map[val]
                ngram_size += 1
                ngram_item += skip_distance
            ngram_start += 1
        if start_ngram_size == 1:
            start_ngram_size += 1
            if start_ngram_size > max_gram_length:
                break