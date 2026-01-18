import struct
from gzip import GzipFile
from io import BytesIO
from scrapy.http import Response
from ._compression import _CHUNK_SIZE, _DecompressionMaxSizeExceeded
def gunzip(data: bytes, *, max_size: int=0) -> bytes:
    """Gunzip the given data and return as much data as possible.

    This is resilient to CRC checksum errors.
    """
    f = GzipFile(fileobj=BytesIO(data))
    output_stream = BytesIO()
    chunk = b'.'
    decompressed_size = 0
    while chunk:
        try:
            chunk = f.read1(_CHUNK_SIZE)
        except (OSError, EOFError, struct.error):
            if output_stream.getbuffer().nbytes > 0:
                break
            raise
        decompressed_size += len(chunk)
        if max_size and decompressed_size > max_size:
            raise _DecompressionMaxSizeExceeded(f'The number of bytes decompressed so far ({decompressed_size} B) exceed the specified maximum ({max_size} B).')
        output_stream.write(chunk)
    output_stream.seek(0)
    return output_stream.read()