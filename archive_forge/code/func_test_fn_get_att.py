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
def test_fn_get_att(self):
    self._create_stack(self.template)
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    mock_sd = {'outputs': [{'name': 'failed', 'error_output': True}, {'name': 'foo'}], 'output_values': {'foo': 'bar', 'deploy_stdout': 'A thing happened', 'deploy_stderr': 'Extraneous logging', 'deploy_status_code': 0}, 'status': self.deployment.COMPLETE}
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.assertEqual('bar', self.deployment.FnGetAtt('foo'))
    self.assertEqual('A thing happened', self.deployment.FnGetAtt('deploy_stdout'))
    self.assertEqual('Extraneous logging', self.deployment.FnGetAtt('deploy_stderr'))
    self.assertEqual(0, self.deployment.FnGetAtt('deploy_status_code'))