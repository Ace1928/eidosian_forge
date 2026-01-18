import errno
import re
import socket
import sys
def apop(self, user, password):
    """Authorisation

        - only possible if server has supplied a timestamp in initial greeting.

        Args:
                user     - mailbox user;
                password - mailbox password.

        NB: mailbox is locked by server from here to 'quit()'
        """
    secret = bytes(password, self.encoding)
    m = self.timestamp.match(self.welcome)
    if not m:
        raise error_proto('-ERR APOP not supported by server')
    import hashlib
    digest = m.group(1) + secret
    digest = hashlib.md5(digest).hexdigest()
    return self._shortcmd('APOP %s %s' % (user, digest))