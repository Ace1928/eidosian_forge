import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def node_list(self, fields=None, params=''):
    """List baremetal nodes.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON node objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal node list {0} {1}'.format(opts, params))
    return json.loads(output)