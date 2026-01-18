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
def test_handle_status_code_failed(self):
    self._create_stack(self.template)
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    rpcc = self.rpc_client
    rpcc.signal_software_deployment.return_value = 'deployment failed'
    details = {'deploy_stdout': 'A thing happened', 'deploy_stderr': 'Then it broke', 'deploy_status_code': -1}
    self.deployment.handle_signal(details)
    ca = rpcc.signal_software_deployment.call_args[0]
    self.assertEqual(self.ctx, ca[0])
    self.assertEqual('c8a19429-7fde-47ea-a42f-40045488226c', ca[1])
    self.assertEqual(details, ca[2])
    self.assertIsNotNone(ca[3])