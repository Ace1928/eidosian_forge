from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
def message_from_headers(hdr):
    bs = '\r\n'.join(('{}: {}'.format(*h) for h in hdr))
    return message_from_bytes(bs.encode())