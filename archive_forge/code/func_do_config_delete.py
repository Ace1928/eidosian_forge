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
@utils.arg('id', metavar='<ID>', nargs='+', help=_('ID of the configuration(s) to delete.'))
def do_config_delete(hc, args):
    """Delete the software configuration(s)."""
    show_deprecated('heat config-delete', 'openstack software config delete')
    failure_count = 0
    for config_id in args.id:
        try:
            hc.software_configs.delete(config_id=config_id)
        except exc.HTTPNotFound:
            failure_count += 1
            print(_('Software config with ID %s not found') % config_id)
    if failure_count:
        raise exc.CommandError(_('Unable to delete %(count)d of the %(total)d configs.') % {'count': failure_count, 'total': len(args.id)})