import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_list(self, fields=None, params=''):
    """List baremetal ports.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON port objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal port list {0} {1}'.format(opts, params))
    return json.loads(output)