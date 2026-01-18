import smtplib
import ssl
import threading
from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.message import sanitize_address
from django.core.mail.utils import DNS_NAME
from django.utils.functional import cached_property
@cached_property
def ssl_context(self):
    if self.ssl_certfile or self.ssl_keyfile:
        ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
        return ssl_context
    else:
        return ssl.create_default_context()