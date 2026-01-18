from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def setup_na_ontap_zapi(module, vserver=None):
    hostname = module.params['hostname']
    username = module.params['username']
    password = module.params['password']
    https = module.params['https']
    validate_certs = module.params['validate_certs']
    port = module.params['http_port']
    version = module.params['ontapi']
    if HAS_NETAPP_LIB:
        server = zapi.NaServer(hostname)
        server.set_username(username)
        server.set_password(password)
        if vserver:
            server.set_vserver(vserver)
        if version:
            minor = version
        else:
            minor = 110
        server.set_api_version(major=1, minor=minor)
        if https:
            if port is None:
                port = 443
            transport_type = 'HTTPS'
            if validate_certs is False:
                if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                    ssl._create_default_https_context = ssl._create_unverified_context
        else:
            if port is None:
                port = 80
            transport_type = 'HTTP'
        server.set_transport_type(transport_type)
        server.set_port(port)
        server.set_server_type('FILER')
        return server
    else:
        module.fail_json(msg='the python NetApp-Lib module is required')