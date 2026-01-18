from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import messaging
from heat.rpc import api as rpc_api
@staticmethod
def make_msg(method, **kwargs):
    return (method, kwargs)