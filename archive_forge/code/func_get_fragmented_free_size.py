def get_fragmented_free_size(self):
    """Returns the amount of space unused, not including the final
        free block.

        :rtype: int
        """
    if not self.starts:
        return 0
    total_free = 0
    free_start = self.starts[0] + self.sizes[0]
    for i, (alloc_start, alloc_size) in enumerate(zip(self.starts[1:], self.sizes[1:])):
        total_free += alloc_start - free_start
        free_start = alloc_start + alloc_size
    return total_free