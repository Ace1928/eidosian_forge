import struct
from llvmlite.ir._utils import _StrCaching
@property
def is_opaque(self):
    return self.elements is None