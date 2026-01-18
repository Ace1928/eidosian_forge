import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def connect_to_origin(self):
    config_stack = config.GlobalStack()
    cert_reqs = config_stack.get('ssl.cert_reqs')
    if self.proxied_host is not None:
        host = self.proxied_host.split(':', 1)[0]
    else:
        host = self.host
    if cert_reqs == ssl.CERT_NONE:
        ui.ui_factory.show_user_warning('not_checking_ssl_cert', host=host)
        ui.ui_factory.suppressed_warnings.add('not_checking_ssl_cert')
        ca_certs = None
    else:
        if self.ca_certs is None:
            ca_certs = config_stack.get('ssl.ca_certs')
        else:
            ca_certs = self.ca_certs
        if ca_certs is None:
            trace.warning("No valid trusted SSL CA certificates file set. See 'brz help ssl.ca_certs' for more information on setting trusted CAs.")
    try:
        ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=ca_certs)
        ssl_context.check_hostname = cert_reqs != ssl.CERT_NONE
        if self.cert_file:
            ssl_context.load_cert_chain(keyfile=self.key_file, certfile=self.cert_file)
        ssl_context.verify_mode = cert_reqs
        ssl_sock = ssl_context.wrap_socket(self.sock, server_hostname=self.host)
    except ssl.SSLError:
        trace.note('\nSee `brz help ssl.ca_certs` for how to specify trusted CAcertificates.\nPass -Ossl.cert_reqs=none to disable certificate verification entirely.\n')
        raise
    self._wrap_socket_for_reporting(ssl_sock)