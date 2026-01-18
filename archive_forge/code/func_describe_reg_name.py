from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_reg_name(regnum, machine_arch=None, default=True):
    """ Provide a textual description for a register name, given its serial
        number. The number is expected to be valid.
    """
    if machine_arch is None:
        machine_arch = _MACHINE_ARCH
    if machine_arch == 'x86':
        return _REG_NAMES_x86[regnum]
    elif machine_arch == 'x64':
        return _REG_NAMES_x64[regnum]
    elif machine_arch == 'AArch64':
        return _REG_NAMES_AArch64[regnum]
    elif default:
        return 'r%s' % regnum
    else:
        return None