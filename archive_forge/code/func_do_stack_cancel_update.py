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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to cancel update for.'))
def do_stack_cancel_update(hc, args):
    """Cancel currently running update of the stack."""
    show_deprecated('heat stack-cancel-update', 'openstack stack cancel')
    fields = {'stack_id': args.id}
    try:
        hc.actions.cancel_update(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % args.id)
    else:
        do_stack_list(hc)