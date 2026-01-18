from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def iter_subsections(self, vendor_name=None):
    """ Yield all subsections (limit to |vendor_name| if specified).
        """
    for subsec in self._make_subsections():
        if vendor_name is None or subsec['vendor_name'] == vendor_name:
            yield subsec