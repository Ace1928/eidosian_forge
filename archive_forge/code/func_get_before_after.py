from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_before_after(self):
    """
        Return texts ``before`` and ``after``.
        """
    before = dict()
    after = dict()
    for item in self._diff:
        before[item['name']] = item['active']
        after[item['name']] = item['parameter']
    return (before, after)