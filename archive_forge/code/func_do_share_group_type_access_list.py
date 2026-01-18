from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('share_group_type', metavar='<share_group_type>', help='Filter results by share group type name or ID.')
def do_share_group_type_access_list(cs, args):
    """Print access information about a share group type (Admin only)."""
    share_group_type = _find_share_group_type(cs, args.share_group_type)
    if share_group_type.is_public:
        raise exceptions.CommandError('Forbidden to get access list for public share group type.')
    access_list = cs.share_group_type_access.list(share_group_type)
    columns = ['Project_ID']
    cliutils.print_list(access_list, columns)