from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def get_full_path(self):
    """ Return the full path filename for the DIE.

            The filename is the join of 'DW_AT_comp_dir' and 'DW_AT_name',
            either of which may be missing in practice. Note that its value is
            usually a string taken from the .debug_string section and the
            returned value will be a string.
        """
    comp_dir_attr = self.attributes.get('DW_AT_comp_dir', None)
    comp_dir = bytes2str(comp_dir_attr.value) if comp_dir_attr else ''
    fname_attr = self.attributes.get('DW_AT_name', None)
    fname = bytes2str(fname_attr.value) if fname_attr else ''
    return os.path.join(comp_dir, fname)