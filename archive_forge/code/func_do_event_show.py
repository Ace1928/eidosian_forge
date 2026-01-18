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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to show the events for.'))
@utils.arg('resource', metavar='<RESOURCE>', help=_('Name of the resource the event belongs to.'))
@utils.arg('event', metavar='<EVENT>', help=_('ID of event to display details for.'))
def do_event_show(hc, args):
    """Describe the event."""
    show_deprecated('heat event-show', 'openstack stack event show')
    fields = {'stack_id': args.id, 'resource_name': args.resource, 'event_id': args.event}
    try:
        event = hc.events.get(**fields)
    except exc.HTTPNotFound as ex:
        raise exc.CommandError(str(ex))
    else:
        formatters = {'links': utils.link_formatter, 'resource_properties': utils.json_formatter}
        utils.print_dict(event.to_dict(), formatters=formatters)