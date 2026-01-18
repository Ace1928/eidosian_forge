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
def check_required_plugins(self):
    missing_plugins = []
    for plugin, params in self.required_plugins:
        for param in params:
            if (param in self.foreman_params or param == '*') and (not self.has_plugin(plugin)):
                if param == '*':
                    param = 'the whole module'
                missing_plugins.append('{0} (for {1})'.format(plugin, param))
    if missing_plugins:
        missing_msg = 'The server is missing required plugins: {0}.'.format(', '.join(missing_plugins))
        self.fail_json(msg=missing_msg)