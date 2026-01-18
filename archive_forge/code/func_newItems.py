from functools import total_ordering
def newItems():
    for i in range(origLen + 1):
        if i == start:
            yield from valueList
        if i < origLen:
            if i < start or i >= stop:
                yield self._get_single_internal(i)