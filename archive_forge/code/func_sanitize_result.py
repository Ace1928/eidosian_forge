from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def sanitize_result(data):
    """Sanitize data object for return to Ansible.

    When the data object contains types such as docker.types.containers.HostConfig,
    Ansible will fail when these are returned via exit_json or fail_json.
    HostConfig is derived from dict, but its constructor requires additional
    arguments. This function sanitizes data structures by recursively converting
    everything derived from dict to dict and everything derived from list (and tuple)
    to a list.
    """
    if isinstance(data, dict):
        return dict(((k, sanitize_result(v)) for k, v in data.items()))
    elif isinstance(data, (list, tuple)):
        return [sanitize_result(v) for v in data]
    else:
        return data