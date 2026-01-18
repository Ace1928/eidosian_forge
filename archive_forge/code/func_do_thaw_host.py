import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('host', metavar='<hostname>', help='Host name.')
def do_thaw_host(cs, args):
    """Thaw and enable the specified cinder-volume host."""
    cs.services.thaw_host(args.host)