import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def retrbinary(self, cmd, callback, blocksize=8192, rest=None):
    """Retrieve data in binary mode.  A new port is created for you.

        Args:
          cmd: A RETR command.
          callback: A single parameter callable to be called on each
                    block of data read.
          blocksize: The maximum number of bytes to read from the
                     socket at one time.  [default: 8192]
          rest: Passed to transfercmd().  [default: None]

        Returns:
          The response code.
        """
    self.voidcmd('TYPE I')
    with self.transfercmd(cmd, rest) as conn:
        while 1:
            data = conn.recv(blocksize)
            if not data:
                break
            callback(data)
        if _SSLSocket is not None and isinstance(conn, _SSLSocket):
            conn.unwrap()
    return self.voidresp()