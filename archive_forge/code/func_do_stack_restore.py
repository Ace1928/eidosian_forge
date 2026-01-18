import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of the stack containing the snapshot.'))
@utils.arg('snapshot', metavar='<SNAPSHOT>', help=_('The ID of the snapshot to restore.'))
def do_stack_restore(hc, args):
    """Restore a snapshot of a stack."""
    show_deprecated('heat stack-restore', 'openstack stack snapshot restore')
    fields = {'stack_id': args.id, 'snapshot_id': args.snapshot}
    try:
        hc.stacks.restore(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack or snapshot not found'))