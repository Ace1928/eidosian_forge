import time
from .exceptions import EOF, TIMEOUT
def existing_data(self):
    spawn = self.spawn
    before_len = spawn._before.tell()
    buf_len = spawn._buffer.tell()
    freshlen = before_len
    if before_len > buf_len:
        if not self.searchwindowsize:
            spawn._buffer = spawn.buffer_type()
            window = spawn._before.getvalue()
            spawn._buffer.write(window)
        elif buf_len < self.searchwindowsize:
            spawn._buffer = spawn.buffer_type()
            spawn._before.seek(max(0, before_len - self.searchwindowsize))
            window = spawn._before.read()
            spawn._buffer.write(window)
        else:
            spawn._buffer.seek(max(0, buf_len - self.searchwindowsize))
            window = spawn._buffer.read()
    elif self.searchwindowsize:
        spawn._buffer.seek(max(0, buf_len - self.searchwindowsize))
        window = spawn._buffer.read()
    else:
        window = spawn._buffer.getvalue()
    return self.do_search(window, freshlen)