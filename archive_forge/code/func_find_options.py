import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def find_options(cls, dct):
    """Returns a new dict with all the items that are a mapping to an
        ``Option``.
        """
    return {k: v for k, v in dct.items() if isinstance(v, Option)}