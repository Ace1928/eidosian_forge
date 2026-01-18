def realloc(self, start, size, new_size):
    """Reallocate a region of the buffer.

        This is more efficient than separate `dealloc` and `alloc` calls, as
        the region can often be resized in-place.

        Raises `AllocatorMemoryException` if the allocation cannot be
        fulfilled.

        :Parameters:
            `start` : int
                Current starting index of the region.
            `size` : int
                Current size of the region.
            `new_size` : int
                New size of the region.

        """
    assert size >= 0 and new_size >= 0
    if new_size == 0:
        if size != 0:
            self.dealloc(start, size)
        return 0
    elif size == 0:
        return self.alloc(new_size)
    if new_size < size:
        self.dealloc(start + new_size, size - new_size)
        return start
    for i, (alloc_start, alloc_size) in enumerate(zip(*(self.starts, self.sizes))):
        p = start - alloc_start
        if p >= 0 and size <= alloc_size - p:
            break
    if not (p >= 0 and size <= alloc_size - p):
        print(list(zip(self.starts, self.sizes)))
        print(start, size, new_size)
        print(p, alloc_start, alloc_size)
    assert p >= 0 and size <= alloc_size - p, 'Region not allocated'
    if size == alloc_size - p:
        is_final_block = i == len(self.starts) - 1
        if not is_final_block:
            free_size = self.starts[i + 1] - (start + size)
        else:
            free_size = self.capacity - (start + size)
        if free_size == new_size - size and (not is_final_block):
            self.sizes[i] += free_size + self.sizes[i + 1]
            del self.starts[i + 1]
            del self.sizes[i + 1]
            return start
        elif free_size > new_size - size:
            self.sizes[i] += new_size - size
            return start
    result = self.alloc(new_size)
    self.dealloc(start, size)
    return result