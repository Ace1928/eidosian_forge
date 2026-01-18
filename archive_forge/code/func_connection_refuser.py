import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def connection_refuser(host):
    raise OSError(errno.ECONNREFUSED, 'Connection Refused')