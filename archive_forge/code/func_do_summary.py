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
@api_versions.wraps('3.12')
@utils.arg('--all-tenants', dest='all_tenants', metavar='<0|1>', nargs='?', type=int, const=1, default=utils.env('ALL_TENANTS', default=0), help='Shows details for all tenants. Admin only.')
def do_summary(cs, args):
    """Get volumes summary."""
    all_tenants = args.all_tenants
    info = cs.volumes.summary(all_tenants)
    formatters = ['total_size', 'total_count']
    if cs.api_version >= api_versions.APIVersion('3.36'):
        formatters.append('metadata')
    shell_utils.print_dict(info['volume-summary'], formatters=formatters)