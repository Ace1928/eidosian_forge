import io
import sys
from gunicorn.http.errors import (NoMoreData, ChunkMissingTerminator,
def parse_trailers(self, unreader, data):
    buf = io.BytesIO()
    buf.write(data)
    idx = buf.getvalue().find(b'\r\n\r\n')
    done = buf.getvalue()[:2] == b'\r\n'
    while idx < 0 and (not done):
        self.get_data(unreader, buf)
        idx = buf.getvalue().find(b'\r\n\r\n')
        done = buf.getvalue()[:2] == b'\r\n'
    if done:
        unreader.unread(buf.getvalue()[2:])
        return b''
    self.req.trailers = self.req.parse_headers(buf.getvalue()[:idx])
    unreader.unread(buf.getvalue()[idx + 4:])