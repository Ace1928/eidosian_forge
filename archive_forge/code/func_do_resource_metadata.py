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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to show the resource metadata for.'))
@utils.arg('resource', metavar='<RESOURCE>', help=_('Name of the resource to show the metadata for.'))
def do_resource_metadata(hc, args):
    """List resource metadata."""
    show_deprecated('heat resource-metadata', 'openstack stack resource metadata')
    fields = {'stack_id': args.id, 'resource_name': args.resource}
    try:
        metadata = hc.resources.metadata(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack or resource not found: %(id)s %(resource)s') % {'id': args.id, 'resource': args.resource})
    else:
        print(jsonutils.dumps(metadata, indent=2))