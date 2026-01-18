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
@utils.arg('container', metavar='<container>', help='ID or name of the container to attach network.')
@utils.arg('--network', metavar='<network>', help='The neutron network that container will attach to.')
@utils.arg('--port', metavar='<port>', help='The neutron port that container will attach to.')
@utils.arg('--fixed-ip', metavar='<fixed_ip>', help='The fixed-ip that container will attach to.')
def do_network_attach(cs, args):
    """Attach a network to the container."""
    opts = {}
    opts['container'] = args.container
    opts['network'] = args.network
    opts['port'] = args.port
    opts['fixed_ip'] = args.fixed_ip
    opts = zun_utils.remove_null_parms(**opts)
    try:
        cs.containers.network_attach(**opts)
        print('Request to attach network to container %s has been accepted.' % args.container)
    except Exception as e:
        print('Attach network to container %(container)s failed: %(e)s' % {'container': args.container, 'e': e})