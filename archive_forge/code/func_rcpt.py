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
def rcpt(self, recip, options=()):
    """SMTP 'rcpt' command -- indicates 1 recipient for this mail."""
    optionlist = ''
    if options and self.does_esmtp:
        optionlist = ' ' + ' '.join(options)
    self.putcmd('rcpt', 'TO:%s%s' % (quoteaddr(recip), optionlist))
    return self.getreply()