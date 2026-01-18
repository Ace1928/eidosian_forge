import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def simple_multipart_message():
    msg = _MULTIPART_HEAD + '--%s--\n' % BOUNDARY
    return msg