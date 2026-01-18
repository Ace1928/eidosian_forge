from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
@property
def data_alignment(self):
    """ Return the logical alignment for this section's data.

        This can be different from the .sh_addralign header field when the
        section is compressed.
        """
    return self._decompressed_align