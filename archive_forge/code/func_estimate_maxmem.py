from __future__ import absolute_import
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.utils import to_bytes
from passlib.utils.compat import PYPY
def estimate_maxmem(n, r, p, fudge=1.05):
    """
    calculate memory required for parameter combination.
    assumes parameters have already been validated.

    .. warning::
        this is derived from OpenSSL's scrypt maxmem formula;
        and may not be correct for other implementations
        (additional buffers, different parallelism tradeoffs, etc).
    """
    maxmem = r * (128 * p + 32 * (n + 2) * UINT32_SIZE)
    maxmem = int(maxmem * fudge)
    return maxmem