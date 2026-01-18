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
@utils.arg('id', metavar='<ID>', nargs='+', help=_('ID of the deployment(s) to delete.'))
def do_deployment_delete(hc, args):
    """Delete the software deployment(s)."""
    show_deprecated('heat deployment-delete', 'openstack software deployment delete')
    failure_count = 0
    for deploy_id in args.id:
        try:
            sd = hc.software_deployments.get(deployment_id=deploy_id)
            hc.software_deployments.delete(deployment_id=deploy_id)
        except Exception as e:
            if isinstance(e, exc.HTTPNotFound):
                print(_('Deployment with ID %s not found') % deploy_id)
            failure_count += 1
            continue
        try:
            config_id = getattr(sd, 'config_id')
            hc.software_configs.delete(config_id=config_id)
        except Exception:
            print(_('Failed to delete the correlative config %(config_id)s of deployment %(deploy_id)s') % {'config_id': config_id, 'deploy_id': deploy_id})
    if failure_count:
        raise exc.CommandError(_('Unable to delete %(count)d of the %(total)d deployments.') % {'count': failure_count, 'total': len(args.id)})