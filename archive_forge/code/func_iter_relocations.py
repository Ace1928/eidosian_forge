from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def iter_relocations(self):
    """ Yield all the relocations in the section
        """
    if self._size == 0:
        return []
    limit = self._offset + self._size
    relr = self._offset
    base = None
    while relr < limit:
        entry = struct_parse(self._relr_struct, self._elffile.stream, stream_pos=relr)
        entry_offset = entry['r_offset']
        if entry_offset & 1 == 0:
            base = entry_offset
            base += self._entrysize
            yield Relocation(entry, self._elffile)
        else:
            elf_assert(base is not None, 'RELR bitmap without base address')
            i = 0
            while True:
                entry_offset = entry_offset >> 1
                if entry_offset == 0:
                    break
                if entry_offset & 1 != 0:
                    calc_offset = base + i * self._entrysize
                    yield Relocation(Container(r_offset=calc_offset), self._elffile)
                i += 1
            base += (8 * self._entrysize - 1) * self._elffile.structs.Elf_addr('').sizeof()
        relr += self._entrysize