import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def keypair_list(self, params=''):
    """Return dictionary with list of keypairs."""
    raw_output = self.openstack('keypair list')
    keypairs = self.parse_show_as_object(raw_output)
    return keypairs