from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import (AnsibleModule, missing_required_lib)
def tcp_statsd_client(**client_params):
    return TCPStatsClient(**client_params)