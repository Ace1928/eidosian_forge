import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def send_email(self, attachment=None, attachment_filename=None, attachment_mime_subtype='plain'):

    class FakeConfig:

        def get(self, option):
            return None
    EmailMessage.send(FakeConfig(), 'from@from.com', 'to@to.com', 'subject', 'body', attachment=attachment, attachment_filename=attachment_filename, attachment_mime_subtype=attachment_mime_subtype)