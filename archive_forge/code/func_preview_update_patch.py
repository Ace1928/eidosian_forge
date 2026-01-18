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
@util.registered_identified_stack
def preview_update_patch(self, req, identity, body):
    """Preview PATCH update for existing stack."""
    data = InstantiationData(body, patch=True)
    args = self.prepare_args(data, is_update=True)
    show_nested = self._param_show_nested(req)
    if show_nested is not None:
        args['show_nested'] = show_nested
    changes = self.rpc_client.preview_update_stack(req.context, identity, data.template(), data.environment(), data.files(), args, environment_files=data.environment_files(), files_container=data.files_container())
    return {'resource_changes': changes}