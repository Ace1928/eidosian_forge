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
@staticmethod
@contextlib.contextmanager
def parse_error_check(data_type):
    try:
        yield
    except ValueError as parse_ex:
        mdict = {'type': data_type, 'error': str(parse_ex)}
        msg = _('%(type)s not in valid format: %(error)s') % mdict
        raise exc.HTTPBadRequest(msg)