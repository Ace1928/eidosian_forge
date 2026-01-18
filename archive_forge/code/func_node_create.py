import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def node_create(self, driver=driver_name, name=None, params=''):
    """Create baremetal node and add cleanup.

        :param String driver: Driver for a new node
        :param String name: Name for a new node
        :param String params: Additional args and kwargs
        :return: JSON object of created node
        """
    if not name:
        name = data_utils.rand_name('baremetal')
    opts = self.get_opts()
    output = self.openstack('baremetal node create {0} --driver {1} --name {2} {3}'.format(opts, driver, name, params))
    node = json.loads(output)
    self.addCleanup(self.node_delete, node['uuid'], True)
    if not output:
        self.fail('Baremetal node has not been created!')
    return node