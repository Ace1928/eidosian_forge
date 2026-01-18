import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def article(self, message_spec=None, *, file=None):
    """Process an ARTICLE command.  Argument:
        - message_spec: article number or message id
        - file: filename string or file object to store the article in
        Returns:
        - resp: server response if successful
        - ArticleInfo: (article number, message id, list of article lines)
        """
    if message_spec is not None:
        cmd = 'ARTICLE {0}'.format(message_spec)
    else:
        cmd = 'ARTICLE'
    return self._artcmd(cmd, file)