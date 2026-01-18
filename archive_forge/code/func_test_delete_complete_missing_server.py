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
def test_delete_complete_missing_server(self):
    """Tests deleting a deployment when the server disappears"""
    self._create_stack(self.template_delete_suspend_resume)
    self.mock_software_config()
    mock_sd = self.mock_deployment()
    mock_sd['server_id'] = 'b509edfb-1448-4b57-8cb1-2e31acccbb8a'
    mock_get_server = self.patchobject(nova.NovaClientPlugin, 'get_server', side_effect=exc.EntityNotFound)
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.rpc_client.update_software_deployment.return_value = mock_sd
    mock_sd['status'] = self.deployment.COMPLETE
    self.assertTrue(self.deployment.check_delete_complete(mock_sd))
    mock_get_server.assert_called_once_with(mock_sd['server_id'])