from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def set_zapi_port_and_transport(server, https, port, validate_certs):
    if https:
        if port is None:
            port = 443
        transport_type = 'HTTPS'
        if validate_certs is False and (not os.environ.get('PYTHONHTTPSVERIFY', '')) and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context
    else:
        if port is None:
            port = 80
        transport_type = 'HTTP'
    server.set_transport_type(transport_type)
    server.set_port(port)