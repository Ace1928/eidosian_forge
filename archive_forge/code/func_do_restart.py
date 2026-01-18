import argparse
from contextlib import closing
import io
import os
import tarfile
import time
import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
@utils.arg('containers', metavar='<container>', nargs='+', help='ID or name of the (container)s to restart.')
@utils.arg('-t', '--timeout', metavar='<timeout>', default=10, help='Seconds to wait for stop before restarting (container)s')
def do_restart(cs, args):
    """Restart specified containers."""
    for container in args.containers:
        try:
            cs.containers.restart(container, args.timeout)
            print('Request to restart container %s has been accepted.' % container)
        except Exception as e:
            print('Restart for container %(container)s failed: %(e)s' % {'container': container, 'e': e})