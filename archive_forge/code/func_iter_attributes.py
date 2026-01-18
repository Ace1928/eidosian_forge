from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def iter_attributes(self, tag=None):
    """ Yield all attributes (limit to |tag| if specified).
        """
    for attribute in self._make_attributes():
        if tag is None or attribute.tag == tag:
            yield attribute