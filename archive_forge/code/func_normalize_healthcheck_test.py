from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def normalize_healthcheck_test(test):
    if isinstance(test, (tuple, list)):
        return [str(e) for e in test]
    return ['CMD-SHELL', str(test)]