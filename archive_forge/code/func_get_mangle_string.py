import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def get_mangle_string(self) -> str:
    """Return a string suitable for symbol mangling.
        """
    zdict = self._make_compression_dictionary()
    comp = zlib.compressobj(zdict=zdict, level=zlib.Z_BEST_COMPRESSION, **self._ZLIB_CONFIG)
    buf = [comp.compress(self.summary().encode())]
    buf.append(comp.flush())
    return base64.b64encode(b''.join(buf)).decode()