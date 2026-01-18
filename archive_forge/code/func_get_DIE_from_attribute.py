from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def get_DIE_from_attribute(self, name):
    """ Return the DIE referenced by the named attribute of this DIE.
            The attribute must be in the reference attribute class.

            name:
                The name of the attribute in the reference class.
        """
    attr = self.attributes[name]
    if attr.form in ('DW_FORM_ref1', 'DW_FORM_ref2', 'DW_FORM_ref4', 'DW_FORM_ref8', 'DW_FORM_ref', 'DW_FORM_ref_udata'):
        refaddr = self.cu.cu_offset + attr.raw_value
        return self.cu.get_DIE_from_refaddr(refaddr)
    elif attr.form in 'DW_FORM_ref_addr':
        return self.cu.dwarfinfo.get_DIE_from_refaddr(attr.raw_value)
    elif attr.form in 'DW_FORM_ref_sig8':
        raise NotImplementedError('%s (type unit by signature)' % attr.form)
    elif attr.form in ('DW_FORM_ref_sup4', 'DW_FORM_ref_sup8', 'DW_FORM_GNU_ref_alt'):
        if self.dwarfinfo.supplementary_dwarfinfo:
            return self.dwarfinfo.supplementary_dwarfinfo.get_DIE_from_refaddr(attr.raw_value)
        raise NotImplementedError('%s to dwo' % attr.form)
    else:
        raise DWARFError('%s is not a reference class form attribute' % attr)