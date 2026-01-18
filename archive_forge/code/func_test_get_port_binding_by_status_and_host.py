import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_get_port_binding_by_status_and_host(self):
    bindings = []
    self.assertIsNone(utils.get_port_binding_by_status_and_host(bindings, constants.INACTIVE))
    bindings.extend([{pb_ext.STATUS: constants.INACTIVE, pb_ext.HOST: 'host-1'}, {pb_ext.STATUS: constants.INACTIVE, pb_ext.HOST: 'host-2'}])
    self.assertEqual('host-1', utils.get_port_binding_by_status_and_host(bindings, constants.INACTIVE)[pb_ext.HOST])
    self.assertEqual('host-2', utils.get_port_binding_by_status_and_host(bindings, constants.INACTIVE, host='host-2')[pb_ext.HOST])
    self.assertIsNone(utils.get_port_binding_by_status_and_host(bindings, constants.ACTIVE))
    self.assertRaises(exceptions.PortBindingNotFound, utils.get_port_binding_by_status_and_host, bindings, constants.ACTIVE, 'host', True, 'port_id')