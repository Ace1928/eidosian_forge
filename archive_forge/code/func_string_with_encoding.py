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
def string_with_encoding(string_):
    """Return a str object together with an encoding.

        :param string\\_: A str or unicode object.
        :return: A tuple (str, encoding), where encoding is one of 'ascii',
            'utf-8', or '8-bit', in that preferred order.
        """
    if isinstance(string_, str):
        try:
            return (string_.encode('ascii'), 'ascii')
        except UnicodeEncodeError:
            return (string_.encode('utf-8'), 'utf-8')
    else:
        try:
            string_.decode('ascii')
            return (string_, 'ascii')
        except UnicodeDecodeError:
            try:
                string_.decode('utf-8')
                return (string_, 'utf-8')
            except UnicodeDecodeError:
                return (string_, '8-bit')