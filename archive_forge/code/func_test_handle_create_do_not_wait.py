import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_handle_create_do_not_wait(self):
    self._create_stack(self.template)
    self.mock_software_config()
    derived_sc = self.mock_derived_software_config()
    self.mock_deployment()
    self.deployment.handle_create()
    self.assertEqual({'action': 'CREATE', 'config_id': derived_sc['id'], 'deployment_id': self.deployment.resource_id, 'input_values': {'foo': 'bar'}, 'server_id': '9f1f0e00-05d2-4ca5-8602-95021f19c9d0', 'stack_user_project_id': '65728b74-cfe7-4f17-9c15-11d4f686e591', 'status': 'IN_PROGRESS', 'status_reason': 'Deploy data available'}, self.rpc_client.create_software_deployment.call_args[1])