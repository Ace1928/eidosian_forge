import struct
from pyglet.image.codecs import ImageDecodeException
def skip_data_sub_blocks(file):
    block_size = read_byte(file)
    while block_size != 0:
        data = file.read(block_size)
        block_size = read_byte(file)