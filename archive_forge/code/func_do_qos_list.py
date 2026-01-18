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
def do_qos_list(cs, args):
    """Lists qos specs."""
    qos_specs = cs.qos_specs.list()
    shell_utils.print_qos_specs_list(qos_specs)