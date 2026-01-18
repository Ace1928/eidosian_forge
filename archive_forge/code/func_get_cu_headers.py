import os
import collections
from collections import OrderedDict
from collections.abc import Mapping
from ..common.utils import struct_parse
from bisect import bisect_right
import math
from ..construct import CString, Struct, If
def get_cu_headers(self):
    """
        Returns all CU headers. Mainly required for readelf.
        """
    if self._cu_headers is None:
        self._entries, self._cu_headers = self._get_entries()
    return self._cu_headers