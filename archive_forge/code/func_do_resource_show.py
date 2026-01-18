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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to show the resource for.'))
@utils.arg('resource', metavar='<RESOURCE>', help=_('Name of the resource to show the details for.'))
@utils.arg('-a', '--with-attr', metavar='<ATTRIBUTE>', help=_('Attribute to show, it can be specified multiple times.'), action='append')
def do_resource_show(hc, args):
    """Describe the resource."""
    show_deprecated('heat resource-show', 'openstack stack resource show')
    fields = {'stack_id': args.id, 'resource_name': args.resource}
    if args.with_attr:
        fields['with_attr'] = list(args.with_attr)
    try:
        resource = hc.resources.get(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack or resource not found: %(id)s %(resource)s') % {'id': args.id, 'resource': args.resource})
    else:
        formatters = {'attributes': utils.json_formatter, 'links': utils.link_formatter, 'required_by': utils.newline_list_formatter}
        utils.print_dict(resource.to_dict(), formatters=formatters)