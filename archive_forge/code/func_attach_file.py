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
def attach_file(self, path, mimetype=None):
    """
        Attach a file from the filesystem.

        Set the mimetype to DEFAULT_ATTACHMENT_MIME_TYPE if it isn't specified
        and cannot be guessed.

        For a text/* mimetype (guessed or specified), decode the file's content
        as UTF-8. If that fails, set the mimetype to
        DEFAULT_ATTACHMENT_MIME_TYPE and don't decode the content.
        """
    path = Path(path)
    with path.open('rb') as file:
        content = file.read()
        self.attach(path.name, content, mimetype)