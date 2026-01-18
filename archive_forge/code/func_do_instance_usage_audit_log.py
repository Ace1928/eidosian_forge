import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('--before', dest='before', metavar='<before>', default=None, help=_('Filters the response by the date and time before which to list usage audits. The date and time stamp format is as follows: CCYY-MM-DD hh:mm:ss.NNNNNN ex 2015-08-27 09:49:58 or 2015-08-27 09:49:58.123456.'))
def do_instance_usage_audit_log(cs, args):
    """List/Get server usage audits."""
    audit_log = cs.instance_usage_audit_log.get(before=args.before).to_dict()
    if 'hosts_not_run' in audit_log:
        audit_log['hosts_not_run'] = pprint.pformat(audit_log['hosts_not_run'])
    utils.print_dict(audit_log)