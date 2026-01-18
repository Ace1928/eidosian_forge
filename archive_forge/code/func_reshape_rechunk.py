from __future__ import annotations
import math
import warnings
from collections import Counter
from functools import reduce
from itertools import product
from operator import mul
import numpy as np
from dask import config
from dask.array.core import Array, normalize_chunks
from dask.array.utils import meta_from_array
from dask.base import tokenize
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M, parse_bytes
def reshape_rechunk(inshape, outshape, inchunks):
    assert all((isinstance(c, tuple) for c in inchunks))
    ii = len(inshape) - 1
    oi = len(outshape) - 1
    result_inchunks = [None for i in range(len(inshape))]
    result_outchunks = [None for i in range(len(outshape))]
    while ii >= 0 or oi >= 0:
        if inshape[ii] == outshape[oi]:
            result_inchunks[ii] = inchunks[ii]
            result_outchunks[oi] = inchunks[ii]
            ii -= 1
            oi -= 1
            continue
        din = inshape[ii]
        dout = outshape[oi]
        if din == 1:
            result_inchunks[ii] = (1,)
            ii -= 1
        elif dout == 1:
            result_outchunks[oi] = (1,)
            oi -= 1
        elif din < dout:
            ileft = ii - 1
            while ileft >= 0 and reduce(mul, inshape[ileft:ii + 1]) < dout:
                ileft -= 1
            if reduce(mul, inshape[ileft:ii + 1]) != dout:
                raise NotImplementedError(_not_implemented_message)
            if all((len(inchunks[i]) == inshape[i] for i in range(ii))):
                for i in range(ii + 1):
                    result_inchunks[i] = inchunks[i]
                result_outchunks[oi] = inchunks[ii] * math.prod(map(len, inchunks[ileft:ii]))
            else:
                for i in range(ileft + 1, ii + 1):
                    result_inchunks[i] = (inshape[i],)
                chunk_reduction = reduce(mul, map(len, inchunks[ileft + 1:ii + 1]))
                result_inchunks[ileft] = expand_tuple(inchunks[ileft], chunk_reduction)
                prod = reduce(mul, inshape[ileft + 1:ii + 1])
                result_outchunks[oi] = tuple((prod * c for c in result_inchunks[ileft]))
            oi -= 1
            ii = ileft - 1
        elif din > dout:
            oleft = oi - 1
            while oleft >= 0 and reduce(mul, outshape[oleft:oi + 1]) < din:
                oleft -= 1
            if reduce(mul, outshape[oleft:oi + 1]) != din:
                raise NotImplementedError(_not_implemented_message)
            cs = reduce(mul, outshape[oleft + 1:oi + 1])
            result_inchunks[ii] = contract_tuple(inchunks[ii], cs)
            for i in range(oleft + 1, oi + 1):
                result_outchunks[i] = (outshape[i],)
            result_outchunks[oleft] = tuple((c // cs for c in result_inchunks[ii]))
            oi = oleft - 1
            ii -= 1
    return (tuple(result_inchunks), tuple(result_outchunks))