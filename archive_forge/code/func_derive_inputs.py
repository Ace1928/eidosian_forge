import copy
import itertools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import output
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import resource_group
from heat.engine.resources import signal_responder
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
def derive_inputs():
    for input_config in inputs:
        value = input_values.pop(input_config.name(), input_config.default())
        yield swc_io.InputConfig(value=value, **input_config.as_dict())
    for inpk, inpv in input_values.items():
        yield swc_io.InputConfig(name=inpk, value=inpv)
    yield swc_io.InputConfig(name=self.DEPLOY_SERVER_ID, value=self.properties[self.SERVER], description=_('ID of the server being deployed to'))
    yield swc_io.InputConfig(name=self.DEPLOY_ACTION, value=action, description=_('Name of the current action being deployed'))
    yield swc_io.InputConfig(name=self.DEPLOY_STACK_ID, value=self.stack.identifier().stack_path(), description=_('ID of the stack this deployment belongs to'))
    yield swc_io.InputConfig(name=self.DEPLOY_RESOURCE_NAME, value=self.name, description=_('Name of this deployment resource in the stack'))
    yield swc_io.InputConfig(name=self.DEPLOY_SIGNAL_TRANSPORT, value=self.properties[self.SIGNAL_TRANSPORT], description=_('How the server should signal to heat with the deployment output values.'))
    if self._signal_transport_cfn():
        yield swc_io.InputConfig(name=self.DEPLOY_SIGNAL_ID, value=self._get_ec2_signed_url(), description=_('ID of signal to use for signaling output values'))
        yield swc_io.InputConfig(name=self.DEPLOY_SIGNAL_VERB, value='POST', description=_('HTTP verb to use for signaling output values'))
    elif self._signal_transport_temp_url():
        yield swc_io.InputConfig(name=self.DEPLOY_SIGNAL_ID, value=self._get_swift_signal_url(), description=_('ID of signal to use for signaling output values'))
        yield swc_io.InputConfig(name=self.DEPLOY_SIGNAL_VERB, value='PUT', description=_('HTTP verb to use for signaling output values'))
    elif self._signal_transport_heat() or self._signal_transport_zaqar():
        creds = self._get_heat_signal_credentials()
        yield swc_io.InputConfig(name=self.DEPLOY_AUTH_URL, value=creds['auth_url'], description=_('URL for API authentication'))
        yield swc_io.InputConfig(name=self.DEPLOY_USERNAME, value=creds['username'], description=_('Username for API authentication'))
        yield swc_io.InputConfig(name=self.DEPLOY_USER_ID, value=creds['user_id'], description=_('User ID for API authentication'))
        yield swc_io.InputConfig(name=self.DEPLOY_PASSWORD, value=creds['password'], description=_('Password for API authentication'))
        yield swc_io.InputConfig(name=self.DEPLOY_PROJECT_ID, value=creds['project_id'], description=_('ID of project for API authentication'))
        if creds['region_name']:
            yield swc_io.InputConfig(name=self.DEPLOY_REGION_NAME, value=creds['region_name'], description=_('Region name for API authentication'))
    if self._signal_transport_zaqar():
        yield swc_io.InputConfig(name=self.DEPLOY_QUEUE_ID, value=self._get_zaqar_signal_queue_id(), description=_('ID of queue to use for signaling output values'))