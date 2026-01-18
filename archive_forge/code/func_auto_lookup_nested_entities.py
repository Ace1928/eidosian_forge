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
def auto_lookup_nested_entities(self):
    for key, entity_spec in self.foreman_spec.items():
        if entity_spec.get('type') in {'nested_list'}:
            for nested_key, nested_spec in entity_spec['foreman_spec'].items():
                for item in self.foreman_params.get(key, []):
                    if nested_key in item and nested_spec.get('resolve', True) and (not _is_resolved(nested_spec, item[nested_key])):
                        item[nested_key] = self._lookup_entity(item[nested_key], nested_spec)