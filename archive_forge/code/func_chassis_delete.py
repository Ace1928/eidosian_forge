import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def chassis_delete(self, uuid, ignore_exceptions=False):
    """Try to delete baremetal chassis by UUID.

        :param String uuid: UUID of the chassis
        :param Bool ignore_exceptions: Ignore exception (needed for cleanUp)
        :return: raw values output
        :raise: CommandFailed exception when command fails to delete a chassis
        """
    try:
        return self.openstack('baremetal chassis delete {0}'.format(uuid))
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise