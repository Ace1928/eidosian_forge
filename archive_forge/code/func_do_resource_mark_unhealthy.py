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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack the resource belongs to.'))
@utils.arg('resource', metavar='<RESOURCE>', help=_('Name or physical ID of the resource.'))
@utils.arg('reason', default='', nargs='?', help=_('Reason for state change.'))
@utils.arg('--reset', default=False, action='store_true', help=_('Set the resource as healthy.'))
def do_resource_mark_unhealthy(hc, args):
    """Set resource's health."""
    show_deprecated('heat resource-mark-unhealthy', 'openstack stack resource mark unhealthy')
    fields = {'stack_id': args.id, 'resource_name': args.resource, 'mark_unhealthy': not args.reset, 'resource_status_reason': args.reason}
    try:
        hc.resources.mark_unhealthy(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack or resource not found: %(id)s %(resource)s') % {'id': args.id, 'resource': args.resource})