import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_group_delete(self, identifier, params='', ignore_exceptions=False):
    """Try to delete baremetal port group by Name or UUID.

        :param String identifier: Name or UUID of the port group
        :param String params: temporary arg to pass api version.
        :param Bool ignore_exceptions: Ignore exception (needed for cleanUp)
        :return: raw values output
        :raise: CommandFailed exception if not ignore_exceptions
        """
    try:
        return self.openstack('baremetal port group delete {0} {1}'.format(identifier, params))
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise