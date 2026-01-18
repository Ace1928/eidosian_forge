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
def parse_cli_args(self):
    """ Command line argument processing """
    parser = argparse.ArgumentParser(description='Produce an Ansible Inventory file based on PyVmomi')
    parser.add_argument('--debug', action='store_true', default=False, help='show debug info')
    parser.add_argument('--list', action='store_true', default=True, help='List instances (default: True)')
    parser.add_argument('--host', action='store', help='Get all the variables about a specific instance')
    parser.add_argument('--refresh-cache', action='store_true', default=False, help='Force refresh of cache by making API requests to VSphere (default: False - use cache files)')
    parser.add_argument('--max-instances', default=None, type=int, help='maximum number of instances to retrieve')
    self.args = parser.parse_args()