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
@utils.arg('volume_type', metavar='<volume_type>', type=str, help='Name or ID of volume type.')
def do_encryption_type_delete(cs, args):
    """Deletes encryption type for a volume type. Admin only."""
    volume_type = shell_utils.find_volume_type(cs, args.volume_type)
    cs.volume_encryption_types.delete(volume_type)