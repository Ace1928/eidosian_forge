import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def send_as_append(_self, msg):
    self.messages.append(msg.as_string(BOUNDARY))