import errno
import re
import socket
import sys
def pass_(self, pswd):
    """Send password, return response

        (response includes message count, mailbox size).

        NB: mailbox is locked by server from here to 'quit()'
        """
    return self._shortcmd('PASS %s' % pswd)