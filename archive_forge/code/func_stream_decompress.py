from __future__ import absolute_import
import struct
import cramjam
def stream_decompress(src, dst, blocksize=_STREAM_TO_STREAM_BLOCK_SIZE, decompressor_cls=StreamDecompressor, start_chunk=None):
    """Takes an incoming file-like object and an outgoing file-like object,
    reads data from src, decompresses it, and writes it to dst. 'src' should
    support the read method, and 'dst' should support the write method.

    The default blocksize is good for almost every scenario.
    :param decompressor_cls: class that implements `decompress` method like
        StreamDecompressor in the module
    :param start_chunk: start block of data that have already been read from
        the input stream (to detect the format, for example)
    """
    decompressor = decompressor_cls()
    while True:
        if start_chunk:
            buf = start_chunk
            start_chunk = None
        else:
            buf = src.read(blocksize)
            if not buf:
                break
        buf = decompressor.decompress(buf)
        if buf:
            dst.write(buf)
    decompressor.flush()