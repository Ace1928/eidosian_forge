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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to get the template for.'))
def do_template_show(hc, args):
    """Get the template for the specified stack."""
    show_deprecated('heat template-show', 'openstack stack template show')
    fields = {'stack_id': args.id}
    try:
        template = hc.stacks.template(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % args.id)
    else:
        if 'heat_template_version' in template:
            print(yaml.safe_dump(template, indent=2))
        else:
            print(jsonutils.dumps(template, indent=2, ensure_ascii=False))