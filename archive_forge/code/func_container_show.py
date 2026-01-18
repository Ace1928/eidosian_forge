from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_show(self, identifier, fields=None, params=''):
    """Show specified container.

        :param String identifier: Name or UUID of the container
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of container
        """
    opts = self.get_opts(fields)
    output = self.openstack('appcontainer show {0} {1} {2}'.format(opts, identifier, params))
    return jsonutils.loads(output)