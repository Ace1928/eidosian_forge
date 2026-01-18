import errno
import re
import socket
import sys
def dele(self, which):
    """Delete message number 'which'.

        Result is 'response'.
        """
    return self._shortcmd('DELE %s' % which)