from __future__ import (absolute_import, division, print_function)
import getpass
import socket
import time
import uuid
from collections import OrderedDict
from contextlib import closing
from os.path import basename
from ansible.errors import AnsibleError, AnsibleRuntimeError
from ansible.module_utils.six import raise_from
from ansible.plugins.callback import CallbackBase
def init_apm_client(self, apm_server_url, apm_service_name, apm_verify_server_cert, apm_secret_token, apm_api_key):
    if apm_server_url:
        return Client(service_name=apm_service_name, server_url=apm_server_url, verify_server_cert=False, secret_token=apm_secret_token, api_key=apm_api_key, use_elastic_traceparent_header=True, debug=True)