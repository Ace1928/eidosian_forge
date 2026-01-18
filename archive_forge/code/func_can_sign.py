from Cryptodome.Util.py3compat import bchr, bord, iter_range
import Cryptodome.Util.number
from Cryptodome.Util.number import (ceil_div,
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
def can_sign(self):
    """Return ``True`` if this object can be used to sign messages."""
    return self._key.has_private()