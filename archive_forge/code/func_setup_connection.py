import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def setup_connection(self):
    import http.client
    proxy = os.environ.get('http_proxy', os.environ.get('HTTP_PROXY', ''))
    if NEOS.scheme == 'https':
        proxy = os.environ.get('https_proxy', os.environ.get('HTTPS_PROXY', proxy))
    if proxy:
        self.transport = ProxiedTransport()
        self.transport.set_proxy(proxy)
    elif NEOS.scheme == 'https':
        self.transport = xmlrpclib.SafeTransport()
    else:
        self.transport = xmlrpclib.Transport()
    self.neos = xmlrpclib.ServerProxy('%s://%s:%s' % (NEOS.scheme, NEOS.host, NEOS.port), transport=self.transport)
    logger.info('Connecting to the NEOS server ... ')
    try:
        result = self.neos.ping()
        logger.info('OK.')
    except (socket.error, xmlrpclib.ProtocolError, http.client.BadStatusLine):
        e = sys.exc_info()[1]
        self.neos = None
        logger.info('Fail: %s' % (e,))
        logger.warning('NEOS is temporarily unavailable:\n\t(%s)' % (e,))