import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def assertMessage(self, expected):
    self.assertLength(1, self.messages)
    self.assertEqualDiff(expected, self.messages[0])