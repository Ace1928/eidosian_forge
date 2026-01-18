from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import messaging
from heat.rpc import api as rpc_api
def error_name_matches(err):
    return self.local_error_name(err) == name