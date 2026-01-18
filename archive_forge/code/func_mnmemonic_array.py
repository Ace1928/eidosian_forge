from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
def mnmemonic_array(self):
    if self.bytecode_array:
        return EHABIBytecodeDecoder(self.bytecode_array).mnemonic_array
    else:
        return None