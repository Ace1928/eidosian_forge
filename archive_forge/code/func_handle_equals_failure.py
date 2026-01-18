from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def handle_equals_failure(module, equals_code, equals_response):
    if equals_code == 400 or equals_code == 500:
        module.fail_json(msg=parse_fail_message(equals_code, equals_response))
    if equals_code == 404 and equals_response['code'] == 'generic_err_command_not_found':
        module.fail_json(msg='Relevant hotfix is not installed on Check Point server. See sk114661 on Check Point Support Center.')