import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
def do_extra_specs_list(cs, args):
    """Lists current volume types and extra specs."""
    vtypes = cs.volume_types.list()
    shell_utils.print_list(vtypes, ['ID', 'Name', 'extra_specs'])