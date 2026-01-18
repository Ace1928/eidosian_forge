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
def test_handle_action_for_component(self):
    self._create_stack(self.template)
    self.mock_software_component()
    mock_sd = self.mock_deployment()
    rsrc = self.stack['deployment_mysql']
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    config_id = '0ff2e903-78d7-4cca-829e-233af3dae705'
    prop_diff = {'config': config_id}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.assertIsNotNone(self.deployment.handle_create())
    self.assertIsNotNone(self.deployment.handle_update(json_snippet=snippet, tmpl_diff=None, prop_diff=prop_diff))
    self.assertIsNotNone(self.deployment.handle_suspend())
    self.assertIsNotNone(self.deployment.handle_resume())
    self.assertIsNotNone(self.deployment.handle_delete())