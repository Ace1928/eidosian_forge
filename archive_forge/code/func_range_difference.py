import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def range_difference(self, other_ranges):
    ans = [self]
    N = len(self.range_lists)
    for other in other_ranges:
        if type(other) is AnyRange:
            return []
        if type(other) is not RangeProduct or len(other.range_lists) != N:
            continue
        tmp = []
        for rp in ans:
            if rp.isdisjoint(other):
                tmp.append(rp)
                continue
            for dim in range(N):
                remainder = []
                for r in rp.range_lists[dim]:
                    remainder.extend(r.range_difference(other.range_lists[dim]))
                if remainder:
                    tmp.append(RangeProduct(list(rp.range_lists)))
                    tmp[-1].range_lists[dim] = remainder
        ans = tmp
    return ans