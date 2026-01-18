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
def create_dummy_stack(self):
    snip = self.stack.t.resource_definitions(self.stack)['deploy_mysql']
    resg = sd.SoftwareDeploymentGroup('test', snip, self.stack)
    resg.resource_id = 'test-test'
    nested = self.patchobject(resg, 'nested')
    nested.return_value = dict(zip(self.server_names, self.servers))
    self._stub_get_attr(resg)
    return resg