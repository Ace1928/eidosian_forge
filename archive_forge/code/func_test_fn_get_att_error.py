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
def test_fn_get_att_error(self):
    self._create_stack(self.template)
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    mock_sd = {'outputs': [], 'output_values': {'foo': 'bar'}}
    self.rpc_client.show_software_deployment.return_value = mock_sd
    err = self.assertRaises(exc.InvalidTemplateAttribute, self.deployment.FnGetAtt, 'foo2')
    self.assertEqual('The Referenced Attribute (deployment_mysql foo2) is incorrect.', str(err))