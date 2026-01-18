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
def test_fn_get_att_convg(self):
    cache_data = {'deployment_mysql': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'foo': 'bar'}})}
    self._create_stack(self.template, cache_data=cache_data)
    self.assertEqual('bar', self.stack.defn[self.deployment.name].FnGetAtt('foo'))