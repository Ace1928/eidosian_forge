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
def putcmd(self, cmd, args=''):
    """Send a command to the server."""
    if args == '':
        s = cmd
    else:
        s = f'{cmd} {args}'
    if '\r' in s or '\n' in s:
        s = s.replace('\n', '\\n').replace('\r', '\\r')
        raise ValueError(f'command and arguments contain prohibited newline characters: {s}')
    self.send(f'{s}{CRLF}')