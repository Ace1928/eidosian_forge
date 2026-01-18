import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def byte_swap_buffer_content(buffer, chunksize, from_endiness, to_endiness):
    """Helper function for byte-swapping the buffers field."""
    to_swap = [buffer.data[i:i + chunksize] for i in range(0, len(buffer.data), chunksize)]
    buffer.data = b''.join([int.from_bytes(byteswap, from_endiness).to_bytes(chunksize, to_endiness) for byteswap in to_swap])