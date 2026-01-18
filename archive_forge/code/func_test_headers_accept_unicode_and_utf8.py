import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_headers_accept_unicode_and_utf8(self):
    for user in ['Pepe Pérez <pperez@ejemplo.com>', 'Pepe PÃ©red <pperez@ejemplo.com>']:
        msg = EmailMessage(user, user, user)
        for header in ['From', 'To', 'Subject']:
            value = msg[header]
            value.encode('ascii')