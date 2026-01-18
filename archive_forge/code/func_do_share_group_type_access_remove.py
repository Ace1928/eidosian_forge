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
@cliutils.arg('share_group_type', metavar='<share_group_type>', help='Share group type name or ID to remove access for the given project.')
@cliutils.arg('project_id', metavar='<project_id>', help='Project ID to remove share group type access for.')
def do_share_group_type_access_remove(cs, args):
    """Removes share group type access for the given project (Admin only)."""
    share_group_type = _find_share_group_type(cs, args.share_group_type)
    cs.share_group_type_access.remove_project_access(share_group_type, args.project_id)