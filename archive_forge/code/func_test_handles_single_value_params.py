from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
def test_handles_single_value_params(self):
    result = util.get_allowed_params(self.params, self.param_types)
    self.assertEqual('foo value', result['foo'])