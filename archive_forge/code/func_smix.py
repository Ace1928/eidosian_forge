import operator
import struct
from passlib.utils.compat import izip
from passlib.crypto.digest import pbkdf2_hmac
from passlib.crypto.scrypt._salsa import salsa20
def smix(self, input):
    """run SCrypt smix function on a single input block

        :arg input:
            byte string containing input data.
            interpreted as 32*r little endian 4 byte integers.

        :returns:
            byte string containing output data
            derived by mixing input using n & r parameters.

        .. note:: time & mem cost are both ``O(n * r)``
        """
    bmix = self.bmix
    bmix_struct = self.bmix_struct
    integerify = self.integerify
    n = self.n
    buffer = list(bmix_struct.unpack(input))

    def vgen():
        i = 0
        while i < n:
            last = tuple(buffer)
            yield last
            bmix(last, buffer)
            i += 1
    V = list(vgen())
    get_v_elem = V.__getitem__
    n_mask = n - 1
    i = 0
    while i < n:
        j = integerify(buffer) & n_mask
        result = tuple((a ^ b for a, b in izip(buffer, get_v_elem(j))))
        bmix(result, buffer)
        i += 1
    return bmix_struct.pack(*buffer)