def heappushpop(heap, item):
    """Fast version of a heappush followed by a heappop."""
    if heap and heap[0] < item:
        item, heap[0] = (heap[0], item)
        _siftup(heap, 0)
    return item