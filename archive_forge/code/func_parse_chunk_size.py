import io
import sys
from gunicorn.http.errors import (NoMoreData, ChunkMissingTerminator,
def parse_chunk_size(self, unreader, data=None):
    buf = io.BytesIO()
    if data is not None:
        buf.write(data)
    idx = buf.getvalue().find(b'\r\n')
    while idx < 0:
        self.get_data(unreader, buf)
        idx = buf.getvalue().find(b'\r\n')
    data = buf.getvalue()
    line, rest_chunk = (data[:idx], data[idx + 2:])
    chunk_size = line.split(b';', 1)[0].strip()
    try:
        chunk_size = int(chunk_size, 16)
    except ValueError:
        raise InvalidChunkSize(chunk_size)
    if chunk_size == 0:
        try:
            self.parse_trailers(unreader, rest_chunk)
        except NoMoreData:
            pass
        return (0, None)
    return (chunk_size, rest_chunk)