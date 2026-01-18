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
def test_check_delete_complete_delete_sd(self):
    self._create_stack(self.template)
    mock_sd = self.mock_deployment()
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.assertTrue(self.deployment.check_delete_complete())
    self.assertEqual((self.ctx, self.deployment.resource_id), self.rpc_client.delete_software_deployment.call_args[0])