import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def set_bookmark(self):
    self._bookmark = self._index