import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def complex_multipart_message(typ):
    msg = _MULTIPART_HEAD + '--%(boundary)s\nMIME-Version: 1.0\nContent-Type: text/%%s; charset="us-ascii"; name="lines.txt"\nContent-Transfer-Encoding: 7bit\nContent-Disposition: inline\n\na\nb\nc\nd\ne\n\n--%(boundary)s--\n' % {'boundary': BOUNDARY}
    return msg % (typ,)