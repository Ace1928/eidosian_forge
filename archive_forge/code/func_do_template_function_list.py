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
@utils.arg('template_version', metavar='<TEMPLATE_VERSION>', help=_('Template version to get the functions for.'))
def do_template_function_list(hc, args):
    """List the available functions."""
    show_deprecated('heat template-function-list', 'openstack orchestration template function list')
    try:
        functions = hc.template_versions.get(args.template_version)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Template version not found: %s') % args.template_version)
    else:
        utils.print_list(functions, ['functions', 'description'])