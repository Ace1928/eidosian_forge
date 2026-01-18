import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def deploy_template_list(self, fields=None, params=''):
    """List baremetal deploy templates.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON deploy template objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal deploy template list {0} {1}'.format(opts, params))
    return json.loads(output)