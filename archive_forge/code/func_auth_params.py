from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
@property
def auth_params(self):
    self.log('Getting credentials')
    client_params = self._get_params()
    params = dict()
    for key in DOCKER_COMMON_ARGS:
        params[key] = client_params.get(key)
    result = dict(docker_host=self._get_value('docker_host', params['docker_host'], 'DOCKER_HOST', DEFAULT_DOCKER_HOST, type='str'), tls_hostname=self._get_value('tls_hostname', params['tls_hostname'], 'DOCKER_TLS_HOSTNAME', None, type='str'), api_version=self._get_value('api_version', params['api_version'], 'DOCKER_API_VERSION', 'auto', type='str'), cacert_path=self._get_value('cacert_path', params['ca_path'], 'DOCKER_CERT_PATH', None, type='str'), cert_path=self._get_value('cert_path', params['client_cert'], 'DOCKER_CERT_PATH', None, type='str'), key_path=self._get_value('key_path', params['client_key'], 'DOCKER_CERT_PATH', None, type='str'), ssl_version=self._get_value('ssl_version', params['ssl_version'], 'DOCKER_SSL_VERSION', None, type='str'), tls=self._get_value('tls', params['tls'], 'DOCKER_TLS', DEFAULT_TLS, type='bool'), tls_verify=self._get_value('tls_verfy', params['validate_certs'], 'DOCKER_TLS_VERIFY', DEFAULT_TLS_VERIFY, type='bool'), timeout=self._get_value('timeout', params['timeout'], 'DOCKER_TIMEOUT', DEFAULT_TIMEOUT_SECONDS, type='int'), use_ssh_client=self._get_value('use_ssh_client', params['use_ssh_client'], None, False, type='bool'))
    update_tls_hostname(result)
    return result