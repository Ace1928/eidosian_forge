from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_ct_show_resource(self):
    ct = self._create_ct(self.t)
    self.ct_mgr.get.return_value = self.fake_ct
    self.assertEqual({'cluster-template': 'info'}, ct.FnGetAtt('show'))
    self.ct_mgr.get.assert_called_once_with('some_ct_id')