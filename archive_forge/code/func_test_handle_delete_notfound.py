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
def test_handle_delete_notfound(self):
    self._create_stack(self.template)
    deployment_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    self.deployment.resource_id = deployment_id
    self.mock_software_config()
    derived_sc = self.mock_derived_software_config()
    mock_sd = self.mock_deployment()
    mock_sd['config_id'] = derived_sc['id']
    self.rpc_client.show_software_deployment.return_value = mock_sd
    nf = exc.NotFound
    self.rpc_client.delete_software_deployment.side_effect = nf
    self.rpc_client.delete_software_config.side_effect = nf
    self.assertIsNone(self.deployment.handle_delete())
    self.assertTrue(self.deployment.check_delete_complete())
    self.assertEqual((self.ctx, derived_sc['id']), self.rpc_client.delete_software_config.call_args[0])