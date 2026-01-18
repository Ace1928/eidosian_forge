from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def lookup_entity(self, key, params=None):
    if key not in self.foreman_params:
        return None
    entity_spec = self.foreman_spec[key]
    if _is_resolved(entity_spec, self.foreman_params[key]):
        return self.foreman_params[key]
    result = self._lookup_entity(self.foreman_params[key], entity_spec, params)
    self.set_entity(key, result)
    return result