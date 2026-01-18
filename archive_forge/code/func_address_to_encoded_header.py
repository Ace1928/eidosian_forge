from email.header import Header
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from . import __version__ as _breezy_version
from .errors import BzrBadParameterNotUnicode
from .osutils import safe_unicode
from .smtp_connection import SMTPConnection
@staticmethod
def address_to_encoded_header(address):
    """RFC2047-encode an address if necessary.

        :param address: An unicode string, or UTF-8 byte string.
        :return: A possibly RFC2047-encoded string.
        """
    if not isinstance(address, str):
        raise BzrBadParameterNotUnicode(address)
    user, email = parseaddr(address)
    if not user:
        return email
    else:
        return formataddr((str(Header(safe_unicode(user))), email))