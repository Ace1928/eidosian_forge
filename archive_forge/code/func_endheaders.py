import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def endheaders(self, message_body=None, *, encode_chunked=False):
    """Indicate that the last header line has been sent to the server.

        This method sends the request to the server.  The optional message_body
        argument can be used to pass a message body associated with the
        request.
        """
    if self.__state == _CS_REQ_STARTED:
        self.__state = _CS_REQ_SENT
    else:
        raise CannotSendHeader()
    self._send_output(message_body, encode_chunked=encode_chunked)