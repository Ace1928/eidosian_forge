import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
def range_intersection(self, other_ranges):
    ans = list(self.range_lists)
    N = len(self.range_lists)
    for other in other_ranges:
        if type(other) is AnyRange:
            continue
        if type(other) is not RangeProduct or len(other.range_lists) != N:
            return []
        for dim in range(N):
            tmp = []
            for r in ans[dim]:
                tmp.extend(r.range_intersection(other.range_lists[dim]))
            if not tmp:
                return []
            ans[dim] = tmp
    return [RangeProduct(ans)]