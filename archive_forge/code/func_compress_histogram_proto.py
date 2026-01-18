import dataclasses
import numpy as np
from typing import Tuple
def compress_histogram_proto(histo, bps=NORMAL_HISTOGRAM_BPS):
    """Creates fixed size histogram by adding compression to accumulated state.

    This routine transforms a histogram at a particular step by interpolating its
    variable number of buckets to represent their cumulative weight at a constant
    number of compression points. This significantly reduces the size of the
    histogram and makes it suitable for a two-dimensional area plot where the
    output of this routine constitutes the ranges for a single x coordinate.

    Args:
      histo: A HistogramProto object.
      bps: Compression points represented in basis points, 1/100ths of a percent.
          Defaults to normal distribution.

    Returns:
      List of values for each basis point.
    """
    if not histo.num:
        return [CompressedHistogramValue(b, 0.0).as_tuple() for b in bps]
    bucket = np.array(histo.bucket)
    bucket_limit = list(histo.bucket_limit)
    weights = (bucket * bps[-1] / (bucket.sum() or 1.0)).cumsum()
    values = []
    j = 0
    while j < len(bps):
        i = np.searchsorted(weights, bps[j], side='right')
        while i < len(weights):
            cumsum = weights[i]
            cumsum_prev = weights[i - 1] if i > 0 else 0.0
            if cumsum == cumsum_prev:
                i += 1
                continue
            if not i or not cumsum_prev:
                lhs = histo.min
            else:
                lhs = max(bucket_limit[i - 1], histo.min)
            rhs = min(bucket_limit[i], histo.max)
            weight = _lerp(bps[j], cumsum_prev, cumsum, lhs, rhs)
            values.append(CompressedHistogramValue(bps[j], weight).as_tuple())
            j += 1
            break
        else:
            break
    while j < len(bps):
        values.append(CompressedHistogramValue(bps[j], histo.max).as_tuple())
        j += 1
    return values