from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def clean_dict_booleans_for_docker_api(data, allow_sequences=False):
    """
    Go doesn't like Python booleans 'True' or 'False', while Ansible is just
    fine with them in YAML. As such, they need to be converted in cases where
    we pass dictionaries to the Docker API (e.g. docker_network's
    driver_options and docker_prune's filters). When `allow_sequences=True`
    YAML sequences (lists, tuples) are converted to [str] instead of str([...])
    which is the expected format of filters which accept lists such as labels.
    """

    def sanitize(value):
        if value is True:
            return 'true'
        elif value is False:
            return 'false'
        else:
            return str(value)
    result = dict()
    if data is not None:
        for k, v in data.items():
            result[str(k)] = [sanitize(e) for e in v] if allow_sequences and is_sequence(v) else sanitize(v)
    return result