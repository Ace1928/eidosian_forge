from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def get_top_DIE(self):
    """ Get the top DIE (which is either a DW_TAG_compile_unit or
            DW_TAG_partial_unit) of this CU
        """
    if len(self._diemap) > 0:
        return self._dielist[0]
    top = DIE(cu=self, stream=self.dwarfinfo.debug_info_sec.stream, offset=self.cu_die_offset)
    self._dielist.insert(0, top)
    self._diemap.insert(0, self.cu_die_offset)
    top._translate_indirect_attributes()
    return top