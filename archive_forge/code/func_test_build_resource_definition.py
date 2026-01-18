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
def test_build_resource_definition(self):
    stack = utils.parse_stack(self.template)
    snip = stack.t.resource_definitions(stack)['deploy_mysql']
    resg = sd.SoftwareDeploymentGroup('test', snip, stack)
    expect = rsrc_defn.ResourceDefinition(None, 'OS::Heat::SoftwareDeployment', {'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'server': 'uuid1', 'signal_transport': 'CFN_SIGNAL'})
    rdef = resg.get_resource_def()
    self.assertEqual(expect, resg.build_resource_definition('server1', rdef))
    rdef = resg.get_resource_def(include_all=True)
    self.assertEqual(expect, resg.build_resource_definition('server1', rdef))