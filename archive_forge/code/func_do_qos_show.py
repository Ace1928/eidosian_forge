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
@utils.arg('qos_specs', metavar='<qos_specs>', help='ID of QoS specifications to show.')
def do_qos_show(cs, args):
    """Shows qos specs details."""
    qos_specs = shell_utils.find_qos_specs(cs, args.qos_specs)
    shell_utils.print_qos_specs(qos_specs)