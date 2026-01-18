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
def add_runas_options(parser):
    """
    Add options for commands which can run tasks as another user

    Note that this includes the options from add_runas_prompt_options().  Only one of these
    functions should be used.
    """
    runas_group = parser.add_argument_group('Privilege Escalation Options', 'control how and which user you become as on target hosts')
    runas_group.add_argument('-b', '--become', default=C.DEFAULT_BECOME, action='store_true', dest='become', help='run operations with become (does not imply password prompting)')
    runas_group.add_argument('--become-method', dest='become_method', default=C.DEFAULT_BECOME_METHOD, help='privilege escalation method to use (default=%s)' % C.DEFAULT_BECOME_METHOD + ', use `ansible-doc -t become -l` to list valid choices.')
    runas_group.add_argument('--become-user', default=None, dest='become_user', type=str, help='run operations as this user (default=%s)' % C.DEFAULT_BECOME_USER)
    parser.add_argument_group(runas_group)
    add_runas_prompt_options(parser)