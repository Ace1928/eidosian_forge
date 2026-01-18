import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def driver_show(self, driver_name, fields=None, params=''):
    """Show specified baremetal driver.

        :param String driver_name: Name of the driver
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of driver
        """
    opts = self.get_opts(fields=fields)
    driver = self.openstack('baremetal driver show {0} {1} {2}'.format(opts, driver_name, params))
    return json.loads(driver)