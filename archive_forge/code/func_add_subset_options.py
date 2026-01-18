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
def add_subset_options(parser):
    """Add options for commands which can run a subset of tasks"""
    parser.add_argument('-t', '--tags', dest='tags', default=C.TAGS_RUN, action='append', help='only run plays and tasks tagged with these values')
    parser.add_argument('--skip-tags', dest='skip_tags', default=C.TAGS_SKIP, action='append', help='only run plays and tasks whose tags do not match these values')