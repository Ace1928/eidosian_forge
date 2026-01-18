from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_rename(self, identifier, name):
    """Rename specified container.

        :param String identifier: Name or UUID of the container
        :param String name: new name for the container
        """
    self.openstack('appcontainer rename {0} {1}'.format(identifier, name))