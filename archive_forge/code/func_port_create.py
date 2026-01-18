import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_create(self, node_id, mac_address=None, params=''):
    """Create baremetal port and add cleanup.

        :param String node_id: baremetal node UUID
        :param String mac_address: MAC address for port
        :param String params: Additional args and kwargs
        :return: JSON object of created port
        """
    if not mac_address:
        mac_address = data_utils.rand_mac_address()
    opts = self.get_opts()
    port = self.openstack('baremetal port create {0} --node {1} {2} {3}'.format(opts, node_id, mac_address, params))
    port = json.loads(port)
    if not port:
        self.fail('Baremetal port has not been created!')
    self.addCleanup(self.port_delete, port['uuid'], True)
    return port