from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_execute(self, identifier, command):
    """Execute in specified container.

        :param String identifier: Name or UUID of the container
        :param String command: command execute in the container
        """
    return self.openstack('appcontainer exec {0} {1}'.format(identifier, command))