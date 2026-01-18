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
@utils.arg('qos_specs', metavar='<qos_specs>', help='ID of QoS specifications.')
def do_qos_get_association(cs, args):
    """Lists all associations for specified qos specs."""
    associations = cs.qos_specs.get_associations(args.qos_specs)
    shell_utils.print_associations_list(associations)