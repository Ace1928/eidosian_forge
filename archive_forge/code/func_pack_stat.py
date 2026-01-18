import binascii
import os
import struct
from .dirstate import DirState, DirstateCorrupt
def pack_stat(st, _b64=binascii.b2a_base64, _pack=struct.Struct('>6L').pack):
    """Convert stat values into a packed representation

    Not all of the fields from the stat included are strictly needed, and by
    just encoding the mtime and mode a slight speed increase could be gained.
    However, using the pyrex version instead is a bigger win.
    """
    return _b64(_pack(st.st_size & 4294967295, int(st.st_mtime) & 4294967295, int(st.st_ctime) & 4294967295, st.st_dev & 4294967295, st.st_ino & 4294967295, st.st_mode))[:-1]