from __future__ import (absolute_import, division, print_function)
import atexit
import datetime
import itertools
import json
import os
import re
import ssl
import sys
import uuid
from time import time
from jinja2 import Environment
from ansible.module_utils.six import integer_types, PY3
from ansible.module_utils.six.moves import configparser
def create_template_mapping(self, inventory, pattern, dtype='string'):
    """ Return a hash of uuid to templated string from pattern """
    mapping = {}
    for k, v in inventory['_meta']['hostvars'].items():
        t = self.env.from_string(pattern)
        newkey = None
        try:
            newkey = t.render(v)
            newkey = newkey.strip()
        except Exception as e:
            self.debugl(e)
        if not newkey:
            continue
        if dtype == 'integer':
            newkey = int(newkey)
        elif dtype == 'boolean':
            if newkey.lower() == 'false':
                newkey = False
            elif newkey.lower() == 'true':
                newkey = True
        elif dtype == 'string':
            pass
        mapping[k] = newkey
    return mapping