import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def authenticate(self, mechanism, authobject):
    """Authenticate command - requires response processing.

        'mechanism' specifies which authentication mechanism is to
        be used - it must appear in <instance>.capabilities in the
        form AUTH=<mechanism>.

        'authobject' must be a callable object:

                data = authobject(response)

        It will be called to process server continuation responses; the
        response argument it is passed will be a bytes.  It should return bytes
        data that will be base64 encoded and sent to the server.  It should
        return None if the client abort response '*' should be sent instead.
        """
    mech = mechanism.upper()
    self.literal = _Authenticator(authobject).process
    typ, dat = self._simple_command('AUTHENTICATE', mech)
    if typ != 'OK':
        raise self.error(dat[-1].decode('utf-8', 'replace'))
    self.state = 'AUTH'
    return (typ, dat)