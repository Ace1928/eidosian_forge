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
def test_handle_create_without_config(self):
    self._create_stack(self.template_no_config)
    self.mock_deployment()
    derived_sc = self.mock_derived_software_config()
    self.deployment.handle_create()
    call_arg = self.rpc_client.create_software_config.call_args[1]
    call_arg['inputs'] = sorted(call_arg['inputs'], key=lambda k: k['name'])
    self.assertEqual({'config': '', 'group': 'Heat::Ungrouped', 'name': self.deployment.physical_resource_name(), 'inputs': [{'name': 'bink', 'type': 'String', 'value': 'bonk'}, {'description': 'Name of the current action being deployed', 'name': 'deploy_action', 'type': 'String', 'value': 'CREATE'}, {'description': 'Name of this deployment resource in the stack', 'name': 'deploy_resource_name', 'type': 'String', 'value': 'deployment_mysql'}, {'description': 'ID of the server being deployed to', 'name': 'deploy_server_id', 'type': 'String', 'value': '9f1f0e00-05d2-4ca5-8602-95021f19c9d0'}, {'description': 'How the server should signal to heat with the deployment output values.', 'name': 'deploy_signal_transport', 'type': 'String', 'value': 'NO_SIGNAL'}, {'description': 'ID of the stack this deployment belongs to', 'name': 'deploy_stack_id', 'type': 'String', 'value': 'software_deployment_test_stack/42f6f66b-631a-44e7-8d01-e22fb54574a9'}, {'name': 'foo', 'type': 'String', 'value': 'bar'}], 'options': None, 'outputs': []}, call_arg)
    self.assertEqual({'action': 'CREATE', 'config_id': derived_sc['id'], 'deployment_id': self.deployment.resource_id, 'input_values': {'bink': 'bonk', 'foo': 'bar'}, 'server_id': '9f1f0e00-05d2-4ca5-8602-95021f19c9d0', 'stack_user_project_id': '65728b74-cfe7-4f17-9c15-11d4f686e591', 'status': 'COMPLETE', 'status_reason': 'Not waiting for outputs signal'}, self.rpc_client.create_software_deployment.call_args[1])