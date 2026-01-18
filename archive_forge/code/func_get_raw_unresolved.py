from collections import defaultdict
import binascii
from io import BytesIO, UnsupportedOperation
from collections import (
import difflib
import struct
from itertools import chain
import os
import sys
from hashlib import sha1
from os import (
from struct import unpack_from
import zlib
from dulwich.errors import (  # noqa: E402
from dulwich.file import GitFile  # noqa: E402
from dulwich.lru_cache import (  # noqa: E402
from dulwich.objects import (  # noqa: E402
def get_raw_unresolved(self, sha1):
    """Get raw unresolved data for a SHA.

        Args:
          sha1: SHA to return data for
        Returns: Tuple with pack object type, delta base (if applicable),
            list of data chunks
        """
    offset = self.index.object_index(sha1)
    obj_type, delta_base, chunks = self.data.get_compressed_data_at(offset)
    if obj_type == OFS_DELTA:
        delta_base = sha_to_hex(self.index.object_sha1(offset - delta_base))
        obj_type = REF_DELTA
    return (obj_type, delta_base, chunks)