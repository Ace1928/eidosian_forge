import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def get_range_list_at_offset(self, offset, cu=None):
    """ Get a range list at the given offset in the section.

            The cu argument is necessary if the ranges section is a
            DWARFv5 debug_rnglists one, and the target rangelist
            contains indirect encodings
        """
    self.stream.seek(offset, os.SEEK_SET)
    return self._parse_range_list_from_stream(cu)