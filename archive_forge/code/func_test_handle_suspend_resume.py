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
def test_handle_suspend_resume(self):
    self._create_stack(self.template_delete_suspend_resume)
    self.mock_software_config()
    derived_sc = self.mock_derived_software_config()
    mock_sd = self.mock_deployment()
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    self.deployment.handle_suspend()
    self.assertEqual({'deployment_id': 'c8a19429-7fde-47ea-a42f-40045488226c', 'action': 'SUSPEND', 'config_id': derived_sc['id'], 'input_values': {'foo': 'bar'}, 'status': 'IN_PROGRESS', 'status_reason': 'Deploy data available'}, self.rpc_client.update_software_deployment.call_args[1])
    mock_sd['status'] = 'IN_PROGRESS'
    self.assertFalse(self.deployment.check_suspend_complete(mock_sd))
    mock_sd['status'] = 'COMPLETE'
    self.assertTrue(self.deployment.check_suspend_complete(mock_sd))
    self.deployment.handle_resume()
    self.assertEqual({'deployment_id': 'c8a19429-7fde-47ea-a42f-40045488226c', 'action': 'RESUME', 'config_id': derived_sc['id'], 'input_values': {'foo': 'bar'}, 'status': 'IN_PROGRESS', 'status_reason': 'Deploy data available'}, self.rpc_client.update_software_deployment.call_args[1])
    mock_sd['status'] = 'IN_PROGRESS'
    self.assertFalse(self.deployment.check_resume_complete(mock_sd))
    mock_sd['status'] = 'COMPLETE'
    self.assertTrue(self.deployment.check_resume_complete(mock_sd))