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
def create_base_parser(prog, usage='', desc=None, epilog=None):
    """
    Create an options parser for all ansible scripts
    """
    parser = ArgumentParser(prog=prog, formatter_class=SortingHelpFormatter, epilog=epilog, description=desc, conflict_handler='resolve')
    version_help = "show program's version number, config file location, configured module search path, module location, executable location and exit"
    parser.add_argument('--version', action=AnsibleVersion, nargs=0, help=version_help)
    add_verbosity_options(parser)
    return parser