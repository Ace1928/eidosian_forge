import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_need_update_for_nested_resource(self):
    """Test the resource with nested stack should need update.

        The resource in Create or Update state and has nested stack, should
        need update.
        """
    self.parent_resource.action = self.parent_resource.CREATE
    self.parent_resource._rpc_client = mock.MagicMock()
    self.parent_resource._rpc_client.show_stack.return_value = [{'stack_action': self.parent_resource.CREATE, 'stack_status': self.parent_resource.COMPLETE}]
    need_update = self.parent_resource._needs_update(self.parent_resource.t, self.parent_resource.t, self.parent_resource.properties, self.parent_resource.properties, self.parent_resource)
    self.assertTrue(need_update)