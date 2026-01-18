import dataclasses
import numpy as np
from typing import Tuple
def compress_histogram(buckets, bps=NORMAL_HISTOGRAM_BPS):
    """Creates fixed size histogram by adding compression to accumulated state.

    This routine transforms a histogram at a particular step by linearly
    interpolating its variable number of buckets to represent their cumulative
    weight at a constant number of compression points. This significantly reduces
    the size of the histogram and makes it suitable for a two-dimensional area
    plot where the output of this routine constitutes the ranges for a single x
    coordinate.

    Args:
      buckets: A list of buckets, each of which is a 3-tuple of the form
        `(min, max, count)`.
      bps: Compression points represented in basis points, 1/100ths of a percent.
          Defaults to normal distribution.

    Returns:
      List of values for each basis point.
    """
    buckets = np.array(buckets)
    if not buckets.size:
        return [CompressedHistogramValue(b, 0.0).as_tuple() for b in bps]
    minmin, maxmax = (buckets[0][0], buckets[-1][1])
    counts = buckets[:, 2]
    right_edges = list(buckets[:, 1])
    weights = (counts * bps[-1] / (counts.sum() or 1.0)).cumsum()
    result = []
    bp_index = 0
    while bp_index < len(bps):
        i = np.searchsorted(weights, bps[bp_index], side='right')
        while i < len(weights):
            cumsum = weights[i]
            cumsum_prev = weights[i - 1] if i > 0 else 0.0
            if cumsum == cumsum_prev:
                i += 1
                continue
            if not i or not cumsum_prev:
                lhs = minmin
            else:
                lhs = max(right_edges[i - 1], minmin)
            rhs = min(right_edges[i], maxmax)
            weight = _lerp(bps[bp_index], cumsum_prev, cumsum, lhs, rhs)
            result.append(CompressedHistogramValue(bps[bp_index], weight).as_tuple())
            bp_index += 1
            break
        else:
            break
    while bp_index < len(bps):
        result.append(CompressedHistogramValue(bps[bp_index], maxmax).as_tuple())
        bp_index += 1
    return result