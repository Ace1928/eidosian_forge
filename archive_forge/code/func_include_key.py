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
def include_key(k):
    return k in keys if keys else True