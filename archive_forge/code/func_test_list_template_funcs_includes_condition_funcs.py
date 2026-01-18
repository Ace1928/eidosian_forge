import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_list_template_funcs_includes_condition_funcs(self, mock_enforce):
    params = {'with_condition_func': 'true'}
    req = self._get('/template_versions/t1/functions', params=params)
    engine_response = [{'functions': 'func1', 'description': 'desc1'}, {'functions': 'condition_func', 'description': 'desc2'}]
    self._test_list_template_functions(mock_enforce, req, engine_response, with_condition=True)