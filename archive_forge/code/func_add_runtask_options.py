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
def add_runtask_options(parser):
    """Add options for commands that run a task"""
    parser.add_argument('-e', '--extra-vars', dest='extra_vars', action='append', type=maybe_unfrack_path('@'), help='set additional variables as key=value or YAML/JSON, if filename prepend with @', default=[])