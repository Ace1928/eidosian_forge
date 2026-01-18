import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def iter_location_lists(self):
    """ Iterates through location lists and view pairs. Returns lists of
        LocationEntry, BaseAddressEntry, and LocationViewPair objects.
        """
    ver5 = self.version >= 5
    stream = self.stream
    stream.seek(0, os.SEEK_END)
    endpos = stream.tell()
    stream.seek(0, os.SEEK_SET)
    all_offsets = set()
    locviews = dict()
    cu_map = dict()
    for cu in self.dwarfinfo.iter_CUs():
        cu_ver = cu['version']
        if (cu_ver >= 5) == ver5:
            for die in cu.iter_DIEs():
                if 'DW_AT_GNU_locviews' in die.attributes:
                    assert 'DW_AT_location' in die.attributes and LocationParser._attribute_has_loc_list(die.attributes['DW_AT_location'], cu_ver)
                    views_offset = die.attributes['DW_AT_GNU_locviews'].value
                    list_offset = die.attributes['DW_AT_location'].value
                    locviews[views_offset] = list_offset
                    cu_map[list_offset] = cu
                    all_offsets.add(views_offset)
                for key in die.attributes:
                    attr = die.attributes[key]
                    if (key != 'DW_AT_location' or 'DW_AT_GNU_locviews' not in die.attributes) and LocationParser.attribute_has_location(attr, cu_ver) and LocationParser._attribute_has_loc_list(attr, cu_ver):
                        list_offset = attr.value
                        all_offsets.add(list_offset)
                        cu_map[list_offset] = cu
    all_offsets = list(all_offsets)
    all_offsets.sort()
    if ver5:
        offset_index = 0
        while stream.tell() < endpos:
            cu_header = struct_parse(self.structs.Dwarf_loclists_CU_header, stream)
            assert cu_header.version == 5
            cu_end_offset = cu_header.offset_after_length + cu_header.unit_length
            while stream.tell() < cu_end_offset:
                next_offset = all_offsets[offset_index]
                if next_offset == stream.tell():
                    locview_pairs = self._parse_locview_pairs(locviews)
                    entries = self._parse_location_list_from_stream_v5(cu_map[stream.tell()])
                    yield (locview_pairs + entries)
                    offset_index += 1
                else:
                    if next_offset > cu_end_offset:
                        next_offset = cu_end_offset
                    stream.seek(next_offset, os.SEEK_SET)
    else:
        for offset in all_offsets:
            list_offset = locviews.get(offset, offset)
            if cu_map[list_offset].header.version < 5:
                stream.seek(offset, os.SEEK_SET)
                locview_pairs = self._parse_locview_pairs(locviews)
                entries = self._parse_location_list_from_stream()
                yield (locview_pairs + entries)