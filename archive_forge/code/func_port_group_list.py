import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_group_list(self, fields=None, params=''):
    """List baremetal port groups.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: JSON object of port group list
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal port group list {0} {1}'.format(opts, params))
    return json.loads(output)