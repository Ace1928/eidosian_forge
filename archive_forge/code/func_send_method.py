import logging
from vine import ensure_promise, promise
from .exceptions import AMQPNotImplementedError, RecoverableConnectionError
from .serialization import dumps, loads
def send_method(self, sig, format=None, args=None, content=None, wait=None, callback=None, returns_tuple=False):
    p = promise()
    conn = self.connection
    if conn is None:
        raise RecoverableConnectionError('connection already closed')
    args = dumps(format, args) if format else ''
    try:
        conn.frame_writer(1, self.channel_id, sig, args, content)
    except StopIteration:
        raise RecoverableConnectionError('connection already closed')
    if callback:
        p.then(callback)
    p()
    if wait:
        return self.wait(wait, returns_tuple=returns_tuple)
    return p