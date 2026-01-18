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
@utils.arg('action', metavar='<action>', choices=['set', 'unset'], help='The action. Valid values are "set" or "unset."')
@utils.arg('metadata', metavar='key=value', nargs='+', default=[], help='Metadata key and value pair to set or unset. For unset, specify only the key.')
def do_qos_key(cs, args):
    """Sets or unsets specifications for a qos spec."""
    keypair = shell_utils.extract_metadata(args)
    if args.action == 'set':
        cs.qos_specs.set_keys(args.qos_specs, keypair)
    elif args.action == 'unset':
        cs.qos_specs.unset_keys(args.qos_specs, list(keypair))