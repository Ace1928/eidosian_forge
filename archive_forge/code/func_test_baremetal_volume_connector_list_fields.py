import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_list_fields(self):
    arglist = ['--fields', 'uuid', 'connector_id']
    verifylist = [('fields', [['uuid', 'connector_id']])]
    fake_vc = copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR)
    fake_vc.pop('type')
    fake_vc.pop('extra')
    fake_vc.pop('node_uuid')
    self.baremetal_mock.volume_connector.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vc, loaded=True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'connector_id')}
    self.baremetal_mock.volume_connector.list.assert_called_with(**kwargs)
    collist = ('UUID', 'Connector ID')
    self.assertEqual(collist, columns)
    datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_volume_connector_connector_id),)
    self.assertEqual(datalist, tuple(data))