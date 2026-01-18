import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_aggregate_refs_map(self):
    resg = self._create_dummy_stack()
    found = resg.FnGetAtt('refs_map')
    expected = {'0': 'ID-0', '1': 'ID-1'}
    self.assertEqual(expected, found)