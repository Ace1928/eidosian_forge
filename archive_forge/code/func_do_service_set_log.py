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
@utils.arg('level', metavar='<log-level>', choices=('INFO', 'WARNING', 'ERROR', 'DEBUG', 'info', 'warning', 'error', 'debug'), help='Desired log level.')
@utils.arg('--binary', choices=('', '*', 'cinder-api', 'cinder-volume', 'cinder-scheduler', 'cinder-backup'), default='', help='Binary to change.')
@utils.arg('--server', default='', help='Host or cluster value for service.')
@utils.arg('--prefix', default='', help='Prefix for the log. ie: "cinder.volume.drivers.".')
def do_service_set_log(cs, args):
    """Sets the service log level."""
    cs.services.set_log_levels(args.level, args.binary, args.server, args.prefix)