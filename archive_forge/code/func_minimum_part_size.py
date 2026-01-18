import hashlib
import math
import binascii
from boto.compat import six
def minimum_part_size(size_in_bytes, default_part_size=DEFAULT_PART_SIZE):
    """Calculate the minimum part size needed for a multipart upload.

    Glacier allows a maximum of 10,000 parts per upload.  It also
    states that the maximum archive size is 10,000 * 4 GB, which means
    the part size can range from 1MB to 4GB (provided it is one 1MB
    multiplied by a power of 2).

    This function will compute what the minimum part size must be in
    order to upload a file of size ``size_in_bytes``.

    It will first check if ``default_part_size`` is sufficient for
    a part size given the ``size_in_bytes``.  If this is not the case,
    then the smallest part size than can accomodate a file of size
    ``size_in_bytes`` will be returned.

    If the file size is greater than the maximum allowed archive
    size of 10,000 * 4GB, a ``ValueError`` will be raised.

    """
    part_size = _MEGABYTE
    if default_part_size * MAXIMUM_NUMBER_OF_PARTS < size_in_bytes:
        if size_in_bytes > 4096 * _MEGABYTE * 10000:
            raise ValueError('File size too large: %s' % size_in_bytes)
        min_part_size = size_in_bytes / 10000
        power = 3
        while part_size < min_part_size:
            part_size = math.ldexp(_MEGABYTE, power)
            power += 1
        part_size = int(part_size)
    else:
        part_size = default_part_size
    return part_size