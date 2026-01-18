import struct
from llvmlite.ir._utils import _StrCaching
@property
def intrinsic_name(self):
    return str(self)