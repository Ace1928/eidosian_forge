from ..construct import CString
from ..common.utils import struct_parse
from .constants import SH_FLAGS
from .notes import iter_notes
def section_in_segment(self, section):
    """ Is the given section contained in this segment?

            Note: this tries to reproduce the intricate rules of the
            ELF_SECTION_IN_SEGMENT_STRICT macro of the header
            elf/include/internal.h in the source of binutils.
        """
    segtype = self['p_type']
    sectype = section['sh_type']
    secflags = section['sh_flags']
    if secflags & SH_FLAGS.SHF_TLS and segtype in ('PT_TLS', 'PT_GNU_RELRO', 'PT_LOAD'):
        pass
    elif secflags & SH_FLAGS.SHF_TLS == 0 and segtype not in ('PT_TLS', 'PT_PHDR'):
        pass
    else:
        return False
    if secflags & SH_FLAGS.SHF_ALLOC == 0 and segtype in ('PT_LOAD', 'PT_DYNAMIC', 'PT_GNU_EH_FRAME', 'PT_GNU_RELRO', 'PT_GNU_STACK'):
        return False
    if secflags & SH_FLAGS.SHF_ALLOC:
        secaddr = section['sh_addr']
        vaddr = self['p_vaddr']
        if not (secaddr >= vaddr and secaddr - vaddr + section['sh_size'] <= self['p_memsz'] and (secaddr - vaddr <= self['p_memsz'] - 1)):
            return False
    if sectype == 'SHT_NOBITS':
        return True
    secoffset = section['sh_offset']
    poffset = self['p_offset']
    return secoffset >= poffset and secoffset - poffset + section['sh_size'] <= self['p_filesz'] and (secoffset - poffset <= self['p_filesz'] - 1)