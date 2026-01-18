import itertools
from webob import exc
from heat.api.openstack.v1 import util
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
@util.registered_identified_stack
def mark_unhealthy(self, req, identity, resource_name, body):
    """Mark a resource as healthy or unhealthy."""
    data = dict()
    VALID_KEYS = RES_UPDATE_MARK_UNHEALTHY, RES_UPDATE_STATUS_REASON = ('mark_unhealthy', rpc_api.RES_STATUS_DATA)
    invalid_keys = set(body) - set(VALID_KEYS)
    if invalid_keys:
        raise exc.HTTPBadRequest(_('Invalid keys in resource mark unhealthy %s') % invalid_keys)
    if RES_UPDATE_MARK_UNHEALTHY not in body:
        raise exc.HTTPBadRequest(_('Missing mandatory (%s) key from mark unhealthy request') % RES_UPDATE_MARK_UNHEALTHY)
    try:
        data[RES_UPDATE_MARK_UNHEALTHY] = param_utils.extract_bool(RES_UPDATE_MARK_UNHEALTHY, body[RES_UPDATE_MARK_UNHEALTHY])
    except ValueError as e:
        raise exc.HTTPBadRequest(str(e))
    data[RES_UPDATE_STATUS_REASON] = body.get(RES_UPDATE_STATUS_REASON, '')
    self.rpc_client.resource_mark_unhealthy(req.context, stack_identity=identity, resource_name=resource_name, **data)