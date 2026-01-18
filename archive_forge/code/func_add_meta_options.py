from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def add_meta_options(parser):
    """Add options for commands which can launch meta tasks from the command line"""
    parser.add_argument('--force-handlers', default=C.DEFAULT_FORCE_HANDLERS, dest='force_handlers', action='store_true', help='run handlers even if a task fails')
    parser.add_argument('--flush-cache', dest='flush_cache', action='store_true', help='clear the fact cache for every host in inventory')