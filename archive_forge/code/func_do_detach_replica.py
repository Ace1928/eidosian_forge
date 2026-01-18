import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
def do_detach_replica(cs, args):
    """Detaches a replica instance from its replication source."""
    instance = _find_instance(cs, args.instance)
    cs.instances.edit(instance, detach_replica_source=True)