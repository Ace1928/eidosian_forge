import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_empty_message(self):
    msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
    self.assertEqualDiff(EMPTY_MESSAGE, msg.as_string())