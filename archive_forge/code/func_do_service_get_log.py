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
@api_versions.wraps('3.32')
@utils.arg('--binary', choices=('', '*', 'cinder-api', 'cinder-volume', 'cinder-scheduler', 'cinder-backup'), default='', help='Binary to query.')
@utils.arg('--server', default='', help='Host or cluster value for service.')
@utils.arg('--prefix', default='', help='Prefix for the log. ie: "sqlalchemy.".')
def do_service_get_log(cs, args):
    """Gets the service log level."""
    log_levels = cs.services.get_log_levels(args.binary, args.server, args.prefix)
    columns = ('Binary', 'Host', 'Prefix', 'Level')
    shell_utils.print_list(log_levels, columns)