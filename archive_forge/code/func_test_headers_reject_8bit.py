import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_headers_reject_8bit(self):
    for i in range(3):
        x = [b'"J. Random Developer" <jrandom@example.com>'] * 3
        x[i] = b'Pepe P\xe9rez <pperez@ejemplo.com>'
        self.assertRaises(BzrBadParameterNotUnicode, EmailMessage, *x)