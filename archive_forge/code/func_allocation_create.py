import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def allocation_create(self, resource_class='allocation-test', params=''):
    opts = self.get_opts()
    output = self.openstack('baremetal allocation create {0} --resource-class {1} {2}'.format(opts, resource_class, params))
    allocation = json.loads(output)
    self.addCleanup(self.allocation_delete, allocation['uuid'], True)
    if not output:
        self.fail('Baremetal allocation has not been created!')
    return allocation