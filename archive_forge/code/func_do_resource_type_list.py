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
@utils.arg('-f', '--filters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Filter parameters to apply on returned resource types. This can be specified multiple times, or once with parameters separated by a semicolon. It can be any of name, version and support_status'), action='append')
def do_resource_type_list(hc, args):
    """List the available resource types."""
    show_deprecated('heat resource-type-list', 'openstack orchestration resource type list')
    types = hc.resource_types.list(filters=utils.format_parameters(args.filters))
    utils.print_list(types, ['resource_type'], sortby_index=0)