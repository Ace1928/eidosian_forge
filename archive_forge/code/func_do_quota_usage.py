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
@utils.arg('tenant', metavar='<tenant_id>', help='ID of tenant for which to list quota usage.')
def do_quota_usage(cs, args):
    """Lists quota usage for a tenant."""
    shell_utils.quota_usage_show(cs.quotas.get(args.tenant, usage=True))