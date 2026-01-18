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
def remove_sensitive_fields(self, entity):
    """ Set fields with 'no_log' option to None """
    if entity:
        for blacklisted_field in self.blacklisted_fields:
            entity[blacklisted_field] = None
    return entity