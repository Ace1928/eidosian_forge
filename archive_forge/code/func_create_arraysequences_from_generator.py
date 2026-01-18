import numbers
from functools import reduce
from operator import mul
import numpy as np
def create_arraysequences_from_generator(gen, n, buffer_sizes=None):
    """Creates :class:`ArraySequence` objects from a generator yielding tuples

    Parameters
    ----------
    gen : generator
        Generator yielding a size `n` tuple containing the values to put in the
        array sequences.
    n : int
        Number of :class:`ArraySequences` object to create.
    buffer_sizes : list of float, optional
        Sizes (in Mb) for each ArraySequence's buffer.
    """
    if buffer_sizes is None:
        buffer_sizes = [4] * n
    seqs = [ArraySequence(buffer_size=size) for size in buffer_sizes]
    for data in gen:
        for i, seq in enumerate(seqs):
            if data[i].nbytes > 0:
                seq.append(data[i], cache_build=True)
    for seq in seqs:
        seq.finalize_append()
    return seqs