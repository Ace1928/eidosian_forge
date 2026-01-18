from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
def populate_grams(els, els_index, n_ngrams: int, ngram_size: int, ngram_id: int, c):
    for _ngrams in range(n_ngrams, 0, -1):
        n = 1
        m = c
        while els_index < len(els):
            p = m.emplace(els[els_index], NgramPart(0))
            if n == ngram_size:
                p.id_ = ngram_id
                ngram_id += 1
                els_index += 1
                break
            if p.empty():
                p.init()
            m = p.leafs_
            n += 1
            els_index += 1
    return ngram_id