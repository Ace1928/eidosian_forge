import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def getreply(self):
    """Get a reply from the server.

        Returns a tuple consisting of:

          - server response code (e.g. '250', or such, if all goes well)
            Note: returns -1 if it can't read response code.

          - server response string corresponding to response code (multiline
            responses are converted to a single, multiline string).

        Raises SMTPServerDisconnected if end-of-file is reached.
        """
    resp = []
    if self.file is None:
        self.file = self.sock.makefile('rb')
    while 1:
        try:
            line = self.file.readline(_MAXLINE + 1)
        except OSError as e:
            self.close()
            raise SMTPServerDisconnected('Connection unexpectedly closed: ' + str(e))
        if not line:
            self.close()
            raise SMTPServerDisconnected('Connection unexpectedly closed')
        if self.debuglevel > 0:
            self._print_debug('reply:', repr(line))
        if len(line) > _MAXLINE:
            self.close()
            raise SMTPResponseException(500, 'Line too long.')
        resp.append(line[4:].strip(b' \t\r\n'))
        code = line[:3]
        try:
            errcode = int(code)
        except ValueError:
            errcode = -1
            break
        if line[3:4] != b'-':
            break
    errmsg = b'\n'.join(resp)
    if self.debuglevel > 0:
        self._print_debug('reply: retcode (%s); Msg: %a' % (errcode, errmsg))
    return (errcode, errmsg)