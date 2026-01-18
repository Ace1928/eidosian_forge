import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
@mock.patch.object(api, 'format_stack_resource')
def test_format_stack_preview(self, mock_fmt_resource):

    def mock_format_resources(res, **kwargs):
        return 'fmt%s' % res
    mock_fmt_resource.side_effect = mock_format_resources
    resources = [1, [2, [3]]]
    self.stack.preview_resources = mock.Mock(return_value=resources)
    stack = api.format_stack_preview(self.stack)
    self.assertIsInstance(stack, dict)
    self.assertIsNone(stack.get('status'))
    self.assertIsNone(stack.get('action'))
    self.assertIsNone(stack.get('status_reason'))
    self.assertEqual('test_stack', stack['stack_name'])
    self.assertIn('resources', stack)
    resources = list(stack['resources'])
    self.assertEqual('fmt1', resources[0])
    resources = list(resources[1])
    self.assertEqual('fmt2', resources[0])
    resources = list(resources[1])
    self.assertEqual('fmt3', resources[0])
    kwargs = mock_fmt_resource.call_args[1]
    self.assertTrue(kwargs['with_props'])