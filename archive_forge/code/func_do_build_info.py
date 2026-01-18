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
def do_build_info(hc, args):
    """Retrieve build information."""
    show_deprecated('heat build-info', 'openstack orchestration build info')
    result = hc.build_info.build_info()
    formatters = {'api': utils.json_formatter, 'engine': utils.json_formatter}
    utils.print_dict(result, formatters=formatters)