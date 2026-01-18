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
@api_versions.wraps('2.66')
@utils.arg('server', metavar='<server>', help=_('Name or UUID of the server to list actions for. Only UUID can be used to list actions on a deleted server.'))
@utils.arg('--marker', dest='marker', metavar='<marker>', default=None, help=_('The last instance action of the previous page; displays list of actions after "marker".'))
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_('Maximum number of instance actions to display. Note that there is a configurable max limit on the server, and the limit that is used will be the minimum of what is requested here and what is configured in the server.'))
@utils.arg('--changes-since', dest='changes_since', metavar='<changes_since>', default=None, help=_('List only instance actions changed later or equal to a certain point of time. The provided time should be an ISO 8061 formatted time. e.g. 2016-03-04T06:27:59Z.'))
@utils.arg('--changes-before', dest='changes_before', metavar='<changes_before>', default=None, help=_('List only instance actions changed earlier or equal to a certain point of time. The provided time should be an ISO 8061 formatted time. e.g. 2016-03-04T06:27:59Z.'), start_version='2.66')
def do_instance_action_list(cs, args):
    """List actions on a server."""
    server = _find_server(cs, args.server, raise_if_notfound=False)
    if args.changes_since:
        try:
            timeutils.parse_isotime(args.changes_since)
        except ValueError:
            raise exceptions.CommandError(_('Invalid changes-since value: %s') % args.changes_since)
    if args.changes_before:
        try:
            timeutils.parse_isotime(args.changes_before)
        except ValueError:
            raise exceptions.CommandError(_('Invalid changes-before value: %s') % args.changes_before)
    actions = cs.instance_action.list(server, marker=args.marker, limit=args.limit, changes_since=args.changes_since, changes_before=args.changes_before)
    utils.print_list(actions, ['Action', 'Request_ID', 'Message', 'Start_Time', 'Updated_At'], sortby_index=3)