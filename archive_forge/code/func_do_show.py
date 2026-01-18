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
@utils.arg('container', metavar='<container>', help='ID or name of the container to show.')
@utils.arg('-f', '--format', metavar='<format>', action='store', choices=['json', 'yaml', 'table'], default='table', help='Print representation of the container.The choices of the output format is json,table,yaml.Defaults to table.')
@utils.arg('--all-projects', action='store_true', default=False, help='Show container(s) in all projects by name.')
def do_show(cs, args):
    """Show details of a container."""
    opts = {}
    opts['id'] = args.container
    opts['all_projects'] = args.all_projects
    opts = zun_utils.remove_null_parms(**opts)
    container = cs.containers.get(**opts)
    if args.format == 'json':
        print(jsonutils.dumps(container._info, indent=4, sort_keys=True))
    elif args.format == 'yaml':
        print(yaml.safe_dump(container._info, default_flow_style=False))
    elif args.format == 'table':
        _show_container(container)