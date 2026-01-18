import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def should_raise(exc):
    with self.rpcapi.ignore_error_by_name('NotSupported'):
        raise exc