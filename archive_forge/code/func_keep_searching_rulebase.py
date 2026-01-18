from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def keep_searching_rulebase(position, current_section, relative_position, relative_position_is_section):
    position_not_found = position is None
    if relative_position_is_section and 'above' not in relative_position:
        relative_section = list(relative_position.values())[0]
        return position_not_found or current_section != relative_section
    return position_not_found