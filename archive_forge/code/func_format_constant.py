import struct
from llvmlite.ir._utils import _StrCaching
def format_constant(self, value):
    itemstring = ', '.join(['{0} {1}'.format(x.type, x.get_reference()) for x in value])
    ret = '{{{0}}}'.format(itemstring)
    return self._wrap_packed(ret)