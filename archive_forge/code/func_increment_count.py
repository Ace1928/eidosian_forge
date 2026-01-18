from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def increment_count(self, ngram_id: int, row_num: int, frequencies: List[int]) -> None:
    ngram_id -= 1
    output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
    frequencies[output_idx] += 1