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
def auto_lookup_entities(self):
    self.auto_lookup_nested_entities()
    return [self.lookup_entity(key) for key, entity_spec in self.foreman_spec.items() if entity_spec.get('resolve', True) and entity_spec.get('type') in {'entity', 'entity_list'}]