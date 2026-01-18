from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def is_checkpoint_param(parameter):
    if parameter == 'auto_publish_session' or parameter == 'state' or parameter == 'wait_for_task' or (parameter == 'wait_for_task_timeout') or (parameter == 'version'):
        return False
    return True