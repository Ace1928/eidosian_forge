import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util
def default_ssl_context_factory():
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=conf.ca_certs)
    context.load_cert_chain(certfile=conf.certfile, keyfile=conf.keyfile)
    context.verify_mode = conf.cert_reqs
    if conf.ciphers:
        context.set_ciphers(conf.ciphers)
    return context