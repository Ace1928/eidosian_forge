from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def read_to_iter(self, reader, read_size=DECOMPRESSION_RECOMMENDED_INPUT_SIZE, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, skip_bytes=0):
    """Read compressed data to an iterator of uncompressed chunks.

        This method will read data from ``reader``, feed it to a decompressor,
        and emit ``bytes`` chunks representing the decompressed result.

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh):
        ...     # Do something with original data.

        ``read_to_iter()`` accepts an object with a ``read(size)`` method that
        will return compressed bytes or an object conforming to the buffer
        protocol.

        ``read_to_iter()`` returns an iterator whose elements are chunks of the
        decompressed data.

        The size of requested ``read()`` from the source can be specified:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, read_size=16384):
        ...    pass

        It is also possible to skip leading bytes in the input data:

        >>> dctx = zstandard.ZstdDecompressor()
        >>> for chunk in dctx.read_to_iter(fh, skip_bytes=1):
        ...    pass

        .. tip::

           Skipping leading bytes is useful if the source data contains extra
           *header* data. Traditionally, you would need to create a slice or
           ``memoryview`` of the data you want to decompress. This would create
           overhead. It is more efficient to pass the offset into this API.

        Similarly to :py:meth:`ZstdCompressor.read_to_iter`, the consumer of the
        iterator controls when data is decompressed. If the iterator isn't consumed,
        decompression is put on hold.

        When ``read_to_iter()`` is passed an object conforming to the buffer protocol,
        the behavior may seem similar to what occurs when the simple decompression
        API is used. However, this API works when the decompressed size is unknown.
        Furthermore, if feeding large inputs, the decompressor will work in chunks
        instead of performing a single operation.

        :param reader:
           Source of compressed data. Can be any object with a
           ``read(size)`` method or any object conforming to the buffer
           protocol.
        :param read_size:
           Integer size of data chunks to read from ``reader`` and feed into
           the decompressor.
        :param write_size:
           Integer size of data chunks to emit from iterator.
        :param skip_bytes:
           Integer number of bytes to skip over before sending data into
           the decompressor.
        :return:
           Iterator of ``bytes`` representing uncompressed data.
        """
    if skip_bytes >= read_size:
        raise ValueError('skip_bytes must be smaller than read_size')
    if hasattr(reader, 'read'):
        have_read = True
    elif hasattr(reader, '__getitem__'):
        have_read = False
        buffer_offset = 0
        size = len(reader)
    else:
        raise ValueError('must pass an object with a read() method or conforms to buffer protocol')
    if skip_bytes:
        if have_read:
            reader.read(skip_bytes)
        else:
            if skip_bytes > size:
                raise ValueError('skip_bytes larger than first input chunk')
            buffer_offset = skip_bytes
    self._ensure_dctx()
    in_buffer = ffi.new('ZSTD_inBuffer *')
    out_buffer = ffi.new('ZSTD_outBuffer *')
    dst_buffer = ffi.new('char[]', write_size)
    out_buffer.dst = dst_buffer
    out_buffer.size = len(dst_buffer)
    out_buffer.pos = 0
    while True:
        assert out_buffer.pos == 0
        if have_read:
            read_result = reader.read(read_size)
        else:
            remaining = size - buffer_offset
            slice_size = min(remaining, read_size)
            read_result = reader[buffer_offset:buffer_offset + slice_size]
            buffer_offset += slice_size
        if not read_result:
            break
        read_buffer = ffi.from_buffer(read_result)
        in_buffer.src = read_buffer
        in_buffer.size = len(read_buffer)
        in_buffer.pos = 0
        while in_buffer.pos < in_buffer.size:
            assert out_buffer.pos == 0
            zresult = lib.ZSTD_decompressStream(self._dctx, out_buffer, in_buffer)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd decompress error: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                data = ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
                out_buffer.pos = 0
                yield data
            if zresult == 0:
                return
        continue