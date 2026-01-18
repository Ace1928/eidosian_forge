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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to describe.'))
@utils.arg('--no-resolve-outputs', action='store_true', help=_('Do not resolve outputs of the stack.'))
def do_stack_show(hc, args):
    """Describe the stack."""
    show_deprecated('heat stack-show', 'openstack stack show')
    fields = {'stack_id': args.id, 'resolve_outputs': not args.no_resolve_outputs}
    _do_stack_show(hc, fields)