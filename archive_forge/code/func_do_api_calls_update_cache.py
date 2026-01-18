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
def do_api_calls_update_cache(self):
    """ Get instances and cache the data """
    self.inventory = self.instances_to_inventory(self.get_instances())
    self.write_to_cache(self.inventory)