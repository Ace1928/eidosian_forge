from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def guess_chunk(shape, maxshape, typesize):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.

    Undocumented and subject to change without warning.
    """
    shape = tuple((x if x != 0 else 1024 for i, x in enumerate(shape)))
    ndims = len(shape)
    if ndims == 0:
        raise ValueError('Chunks not allowed for scalar datasets.')
    chunks = np.array(shape, dtype='=f8')
    if not np.all(np.isfinite(chunks)):
        raise ValueError('Illegal value in chunk tuple')
    dset_size = product(chunks) * typesize
    target_size = CHUNK_BASE * 2 ** np.log10(dset_size / (1024.0 * 1024))
    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN
    idx = 0
    while True:
        chunk_bytes = product(chunks) * typesize
        if (chunk_bytes < target_size or abs(chunk_bytes - target_size) / target_size < 0.5) and chunk_bytes < CHUNK_MAX:
            break
        if product(chunks) == 1:
            break
        chunks[idx % ndims] = np.ceil(chunks[idx % ndims] / 2.0)
        idx += 1
    return tuple((int(x) for x in chunks))