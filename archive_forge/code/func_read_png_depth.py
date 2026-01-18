import binascii
import struct
from typing import Optional
def read_png_depth(filename: str) -> Optional[int]:
    """Read the special tEXt chunk indicating the depth from a PNG file."""
    with open(filename, 'rb') as f:
        f.seek(-(LEN_IEND + LEN_DEPTH), 2)
        depthchunk = f.read(LEN_DEPTH)
        if not depthchunk.startswith(DEPTH_CHUNK_LEN + DEPTH_CHUNK_START):
            return None
        else:
            return struct.unpack('!i', depthchunk[14:18])[0]