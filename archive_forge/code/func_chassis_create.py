import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def chassis_create(self, params=''):
    """Create baremetal chassis and add cleanup.

        :param String params: Additional args and kwargs
        :return: JSON object of created chassis
        """
    opts = self.get_opts()
    chassis = self.openstack('baremetal chassis create {0} {1}'.format(opts, params))
    chassis = json.loads(chassis)
    if not chassis:
        self.fail('Baremetal chassis has not been created!')
    self.addCleanup(self.chassis_delete, chassis['uuid'], True)
    return chassis