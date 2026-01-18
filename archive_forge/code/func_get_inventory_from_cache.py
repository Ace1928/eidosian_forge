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
def get_inventory_from_cache(self):
    """ Read in jsonified inventory """
    jdata = None
    with open(self.cache_path_cache, 'r') as f:
        jdata = f.read()
    return json.loads(jdata)