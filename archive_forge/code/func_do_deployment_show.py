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
@utils.arg('id', metavar='<ID>', help=_('ID of the deployment.'))
def do_deployment_show(hc, args):
    """Show the details of a software deployment."""
    show_deprecated('heat deployment-show', 'openstack software deployment show')
    try:
        sd = hc.software_deployments.get(deployment_id=args.id)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Deployment not found: %s') % args.id)
    else:
        print(jsonutils.dumps(sd.to_dict(), indent=2))