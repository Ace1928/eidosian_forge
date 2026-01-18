from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_handle_adopt(self):
    self.res.create_with_template = mock.Mock(return_value=None)
    self.res.handle_adopt(resource_data={'resource_id': 'fred'})
    self.res.create_with_template.assert_called_once_with(self.nested_parsed, self.nested_params, None, adopt_data={'resource_id': 'fred'})