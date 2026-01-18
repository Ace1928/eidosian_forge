import contextlib
from oslo_log import log as logging
from urllib import parse
from webob import exc
from heat.api.openstack.v1 import util
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import environment_format
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
@util.registered_policy_enforce
def list_resource_types(self, req):
    """Returns a resource types list which may be used in template."""
    support_status = req.params.get('support_status')
    type_name = req.params.get('name')
    version = req.params.get('version')
    if req.params.get('with_description') is not None:
        with_description = self._extract_bool_param('with_description', req.params.get('with_description'))
    else:
        with_description = False
    return {'resource_types': self.rpc_client.list_resource_types(req.context, support_status=support_status, type_name=type_name, heat_version=version, with_description=with_description)}