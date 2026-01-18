import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def hasOnlyInts(self, only_non_negative=True):
    """Return ``True`` if all items in this sequence are integers
                or non-negative integers.

                This function returns False is the sequence is empty,
                or at least one member is not an integer.

                Args:
                  only_non_negative (boolean):
                    If ``True``, the presence of negative integers
                    causes the method to return ``False``."""
    return self._seq and self.hasInts(only_non_negative) == len(self._seq)