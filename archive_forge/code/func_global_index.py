from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import param_utils
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
@util.registered_policy_enforce
def global_index(self, req):
    return self._index(req, use_admin_cnxt=True)