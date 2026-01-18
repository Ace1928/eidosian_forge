import mimetypes
from email import charset as Charset
from email import encoders as Encoders
from email import generator, message_from_string
from email.errors import HeaderParseError
from email.header import Header
from email.headerregistry import Address, parser
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, getaddresses, make_msgid
from io import BytesIO, StringIO
from pathlib import Path
from django.conf import settings
from django.core.mail.utils import DNS_NAME
from django.utils.encoding import force_str, punycode
def sanitize_address(addr, encoding):
    """
    Format a pair of (name, address) or an email address string.
    """
    address = None
    if not isinstance(addr, tuple):
        addr = force_str(addr)
        try:
            token, rest = parser.get_mailbox(addr)
        except (HeaderParseError, ValueError, IndexError):
            raise ValueError('Invalid address "%s"' % addr)
        else:
            if rest:
                raise ValueError('Invalid address; only %s could be parsed from "%s"' % (token, addr))
            nm = token.display_name or ''
            localpart = token.local_part
            domain = token.domain or ''
    else:
        nm, address = addr
        if '@' not in address:
            raise ValueError(f'Invalid address "{address}"')
        localpart, domain = address.rsplit('@', 1)
    address_parts = nm + localpart + domain
    if '\n' in address_parts or '\r' in address_parts:
        raise ValueError('Invalid address; address parts cannot contain newlines.')
    try:
        nm.encode('ascii')
        nm = Header(nm).encode()
    except UnicodeEncodeError:
        nm = Header(nm, encoding).encode()
    try:
        localpart.encode('ascii')
    except UnicodeEncodeError:
        localpart = Header(localpart, encoding).encode()
    domain = punycode(domain)
    parsed_address = Address(username=localpart, domain=domain)
    return formataddr((nm, parsed_address.addr_spec))