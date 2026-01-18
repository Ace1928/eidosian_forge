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
def test_ngt_show_resource(self):
    ngt = self._create_ngt(self.t)
    self.ngt_mgr.get.return_value = self.fake_ngt
    self.assertEqual({'ng-template': 'info'}, ngt.FnGetAtt('show'))
    self.ngt_mgr.get.assert_called_once_with('some_ng_id')