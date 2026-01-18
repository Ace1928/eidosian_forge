import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
@staticmethod
def get_message_addresses(message):
    """Get the origin and destination addresses of a message.

        :param message: A message object supporting get() to access its
            headers, like email.message.Message or
            breezy.email_message.EmailMessage.
        :return: A pair (from_email, to_emails), where from_email is the email
            address in the From header, and to_emails a list of all the
            addresses in the To, Cc, and Bcc headers.
        """
    from_email = parseaddr(message.get('From', None))[1]
    to_full_addresses = []
    for header in ['To', 'Cc', 'Bcc']:
        value = message.get(header, None)
        if value:
            to_full_addresses.append(value)
    to_emails = [pair[1] for pair in getaddresses(to_full_addresses)]
    return (from_email, to_emails)