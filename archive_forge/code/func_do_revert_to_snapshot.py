import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.40')
@utils.arg('snapshot', metavar='<snapshot>', help='Name or ID of the snapshot to restore. The snapshot must be the most recent one known to cinder.')
def do_revert_to_snapshot(cs, args):
    """Revert a volume to the specified snapshot."""
    snapshot = shell_utils.find_volume_snapshot(cs, args.snapshot)
    volume = utils.find_volume(cs, snapshot.volume_id)
    volume.revert_to_snapshot(snapshot)