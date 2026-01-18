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
def list_template_functions(self, req, template_version):
    """Returns a list of available functions in a given template."""
    if req.params.get('with_condition_func') is not None:
        with_condition = self._extract_bool_param('with_condition_func', req.params.get('with_condition_func'))
    else:
        with_condition = False
    return {'template_functions': self.rpc_client.list_template_functions(req.context, template_version, with_condition)}