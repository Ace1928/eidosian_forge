from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import filter, str
from future import utils
import os
import sys
import ssl
import pprint
import socket
from future.backports.urllib import parse as urllib_parse
from future.backports.http.server import (HTTPServer as _HTTPServer,
from future.backports.test import support
def make_https_server(case, certfile=CERTFILE, host=HOST, handler_class=None):
    context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    context.load_cert_chain(certfile)
    server = HTTPSServerThread(context, host, handler_class)
    flag = threading.Event()
    server.start(flag)
    flag.wait()

    def cleanup():
        if support.verbose:
            sys.stdout.write('stopping HTTPS server\n')
        server.stop()
        if support.verbose:
            sys.stdout.write('joining HTTPS thread\n')
        server.join()
    case.addCleanup(cleanup)
    return server