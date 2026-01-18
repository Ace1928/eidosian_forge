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
@api_versions.wraps('3.38')
@utils.arg('group', metavar='<group>', help='Name or ID of the group.')
def do_group_enable_replication(cs, args):
    """Enables replication for group."""
    shell_utils.find_group(cs, args.group).enable_replication()