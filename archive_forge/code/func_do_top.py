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
@utils.arg('container', metavar='<container>', help='ID or name of the container to display processes.')
@utils.arg('--pid', metavar='<pid>', action='append', default=[], help='The args of the ps id.')
def do_top(cs, args):
    """Display the running processes inside the container."""
    if args.pid:
        output = cs.containers.top(args.container, ' '.join(args.pid))
    else:
        output = cs.containers.top(args.container)
    for titles in output['Titles']:
        (print('%-20s') % titles,)
    if output['Processes']:
        for process in output['Processes']:
            print('')
            for info in process:
                (print('%-20s') % info,)
    else:
        print('')