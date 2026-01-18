import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def keypair_delete(self, name, ignore_exceptions=False):
    """Try to delete keypair by name."""
    try:
        self.openstack('keypair delete ' + name)
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise