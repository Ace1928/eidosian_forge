from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_delete(self, identifier, force=True, ignore_exceptions=False):
    """Try to delete container by name or UUID.

        :param String identifier: Name or UUID of the container
        :param Bool ignore_exceptions: Ignore exception (needed for cleanUp)
        :return: raw values output
        :raise: CommandFailed exception when command fails
                to delete a container
        """
    arg = '--force' if force else ''
    try:
        return self.openstack('appcontainer delete {0} {1}'.format(arg, identifier))
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise