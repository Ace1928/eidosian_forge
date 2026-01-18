import errno
import re
import socket
import sys
def retr(self, which):
    """Retrieve whole message number 'which'.

        Result is in form ['response', ['line', ...], octets].
        """
    return self._longcmd('RETR %s' % which)