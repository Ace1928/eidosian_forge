import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.11')
def do_group_specs_list(cs, args):
    """Lists current group types and specs."""
    gtypes = cs.group_types.list()
    shell_utils.print_list(gtypes, ['ID', 'Name', 'group_specs'])