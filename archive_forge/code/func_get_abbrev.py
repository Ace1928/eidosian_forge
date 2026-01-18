from ..common.utils import struct_parse, dwarf_assert
def get_abbrev(self, code):
    """ Get the AbbrevDecl for a given code. Raise KeyError if no
            declaration for this code exists.
        """
    return self._abbrev_map[code]