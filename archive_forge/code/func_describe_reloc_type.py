from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_reloc_type(x, elffile):
    arch = elffile.get_machine_arch()
    if arch == 'x86':
        return _DESCR_RELOC_TYPE_i386.get(x, _unknown)
    elif arch == 'x64':
        return _DESCR_RELOC_TYPE_x64.get(x, _unknown)
    elif arch == 'ARM':
        return _DESCR_RELOC_TYPE_ARM.get(x, _unknown)
    elif arch == 'AArch64':
        return _DESCR_RELOC_TYPE_AARCH64.get(x, _unknown)
    elif arch == '64-bit PowerPC':
        return _DESCR_RELOC_TYPE_PPC64.get(x, _unknown)
    elif arch == 'IBM S/390':
        return _DESCR_RELOC_TYPE_S390X.get(x, _unknown)
    elif arch == 'MIPS':
        return _DESCR_RELOC_TYPE_MIPS.get(x, _unknown)
    elif arch == 'LoongArch':
        return _DESCR_RELOC_TYPE_LOONGARCH.get(x, _unknown)
    else:
        return 'unrecognized: %-7x' % (x & 4294967295)