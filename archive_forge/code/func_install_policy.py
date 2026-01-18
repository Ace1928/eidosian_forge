from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def install_policy(connection, policy_package, targets):
    payload = {'policy-package': policy_package, 'targets': targets}
    connection.send_request('/web_api/install-policy', payload)