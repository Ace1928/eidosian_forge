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
@utils.arg('resource_type', metavar='<RESOURCE_TYPE>', help=_('Resource type to generate a template for.'))
@utils.arg('-t', '--template-type', metavar='<TEMPLATE_TYPE>', default='cfn', help=_('Template type to generate, hot or cfn.'))
@utils.arg('-F', '--format', metavar='<FORMAT>', help=_('The template output format, one of: %s.') % ', '.join(utils.supported_formats.keys()))
def do_resource_type_template(hc, args):
    """Generate a template based on a resource type."""
    show_deprecated('heat resource-type-template', 'openstack orchestration resource type show --template-type hot')
    fields = {'resource_type': args.resource_type, 'template_type': args.template_type}
    try:
        template = hc.resource_types.generate_template(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Resource Type %s not found.') % args.resource_type)
    else:
        if args.format:
            print(utils.format_output(template, format=args.format))
        else:
            print(utils.format_output(template))