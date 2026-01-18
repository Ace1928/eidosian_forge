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
@utils.arg('-i', '--input-value', metavar='<KEY=VALUE>', help=_('Input value to set on the deployment. This can be specified multiple times.'), action='append')
@utils.arg('-a', '--action', metavar='<ACTION>', default='UPDATE', help=_('Name of action for this deployment. Can be a custom action, or one of: CREATE, UPDATE, DELETE, SUSPEND, RESUME'))
@utils.arg('-c', '--config', metavar='<CONFIG>', help=_('ID of the configuration to deploy.'))
@utils.arg('-s', '--server', metavar='<SERVER>', required=True, help=_('ID of the server being deployed to.'))
@utils.arg('-t', '--signal-transport', default='TEMP_URL_SIGNAL', metavar='<TRANSPORT>', help=_('How the server should signal to heat with the deployment output values. TEMP_URL_SIGNAL will create a Swift TempURL to be signaled via HTTP PUT. NO_SIGNAL will result in the resource going to the COMPLETE state without waiting for any signal.'))
@utils.arg('--container', metavar='<CONTAINER_NAME>', help=_('Optional name of container to store TEMP_URL_SIGNAL objects in. If not specified a container will be created with a name derived from the DEPLOY_NAME'))
@utils.arg('--timeout', metavar='<TIMEOUT>', type=int, default=60, help=_('Deployment timeout in minutes.'))
@utils.arg('name', metavar='<DEPLOY_NAME>', help=_('Name of the derived config associated with this deployment. This is used to apply a sort order to the list of configurations currently deployed to the server.'))
def do_deployment_create(hc, args):
    """Create a software deployment."""
    show_deprecated('heat deployment-create', 'openstack software deployment create')
    config = {}
    if args.config:
        try:
            config = hc.software_configs.get(config_id=args.config)
        except exc.HTTPNotFound:
            raise exc.CommandError(_('Configuration not found: %s') % args.config)
    derrived_params = deployment_utils.build_derived_config_params(action=args.action, source=config, name=args.name, input_values=utils.format_parameters(args.input_value, False), server_id=args.server, signal_transport=args.signal_transport, signal_id=deployment_utils.build_signal_id(hc, args))
    derived_config = hc.software_configs.create(**derrived_params)
    sd = hc.software_deployments.create(tenant_id='asdf', config_id=derived_config.id, server_id=args.server, action=args.action, status='IN_PROGRESS')
    print(jsonutils.dumps(sd.to_dict(), indent=2))