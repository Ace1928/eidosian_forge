from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def iter_subsubsections(self, scope=None):
    """ Yield all subsubsections (limit to |scope| if specified).
        """
    for subsubsec in self._make_subsubsections():
        if scope is None or subsubsec.header.tag == scope:
            yield subsubsec