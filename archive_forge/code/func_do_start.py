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
@utils.arg('containers', metavar='<container>', nargs='+', help='ID of the (container)s to start.')
def do_start(cs, args):
    """Start specified containers."""
    for container in args.containers:
        try:
            cs.containers.start(container)
            print('Request to start container %s has been accepted.' % container)
        except Exception as e:
            print('Start for container %(container)s failed: %(e)s' % {'container': container, 'e': e})