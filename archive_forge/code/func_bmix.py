import operator
import struct
from passlib.utils.compat import izip
from passlib.crypto.digest import pbkdf2_hmac
from passlib.crypto.scrypt._salsa import salsa20
def bmix(self, source, target):
    """
        block mixing function used by smix()
        uses salsa20/8 core to mix block contents.

        :arg source:
            source to read from.
            should be list of 32*r 4-byte integers
            (2*r salsa20 blocks).

        :arg target:
            target to write to.
            should be list with same size as source.
            the existing value of this buffer is ignored.

        .. warning::

            this operates *in place* on target,
            so source & target should NOT be same list.

        .. note::

            * time cost is ``O(r)`` -- loops 16*r times, salsa20() has ``O(1)`` cost.

            * memory cost is ``O(1)`` -- salsa20() uses 16 x uint4,
              all other operations done in-place.
        """
    half = self.bmix_half_len
    tmp = source[-16:]
    siter = iter(source)
    j = 0
    while j < half:
        jn = j + 16
        target[j:jn] = tmp = salsa20((a ^ b for a, b in izip(tmp, siter)))
        target[half + j:half + jn] = tmp = salsa20((a ^ b for a, b in izip(tmp, siter)))
        j = jn