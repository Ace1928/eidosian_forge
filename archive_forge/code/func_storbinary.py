import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def storbinary(self, cmd, fp, blocksize=8192, callback=None, rest=None):
    """Store a file in binary mode.  A new port is created for you.

        Args:
          cmd: A STOR command.
          fp: A file-like object with a read(num_bytes) method.
          blocksize: The maximum data size to read from fp and send over
                     the connection at once.  [default: 8192]
          callback: An optional single parameter callable that is called on
                    each block of data after it is sent.  [default: None]
          rest: Passed to transfercmd().  [default: None]

        Returns:
          The response code.
        """
    self.voidcmd('TYPE I')
    with self.transfercmd(cmd, rest) as conn:
        while 1:
            buf = fp.read(blocksize)
            if not buf:
                break
            conn.sendall(buf)
            if callback:
                callback(buf)
        if _SSLSocket is not None and isinstance(conn, _SSLSocket):
            conn.unwrap()
    return self.voidresp()