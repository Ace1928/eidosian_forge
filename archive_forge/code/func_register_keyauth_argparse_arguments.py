import logging
import os
import sys
import warnings
from cliff import app
from cliff import commandmanager
from keystoneauth1 import loading
from oslo_utils import importutils
from vitrageclient import __version__
from vitrageclient import auth
from vitrageclient import client
from vitrageclient.v1.cli import alarm
from vitrageclient.v1.cli import event
from vitrageclient.v1.cli import healthcheck
from vitrageclient.v1.cli import rca
from vitrageclient.v1.cli import resource
from vitrageclient.v1.cli import service
from vitrageclient.v1.cli import status
from vitrageclient.v1.cli import template
from vitrageclient.v1.cli import topology
from vitrageclient.v1.cli import webhook
@staticmethod
def register_keyauth_argparse_arguments(parser):
    parser.add_argument('--os-region-name', metavar='<auth-region-name>', dest='region_name', default=os.environ.get('OS_REGION_NAME'), help='Authentication region name (Env: OS_REGION_NAME)')
    parser.add_argument('--os-interface', metavar='<interface>', dest='interface', choices=['admin', 'public', 'internal'], default=os.environ.get('OS_INTERFACE'), help='Select an interface type. Valid interface types: [admin, public, internal]. (Env: OS_INTERFACE)')
    loading.register_session_argparse_arguments(parser=parser)
    return loading.register_auth_argparse_arguments(parser=parser, argv=sys.argv, default='password')