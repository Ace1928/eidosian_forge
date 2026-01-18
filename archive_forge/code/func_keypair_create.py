import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def keypair_create(self, name=data_utils.rand_uuid()):
    """Create keypair and add cleanup."""
    raw_output = self.openstack('keypair create ' + name)
    self.addCleanup(self.keypair_delete, name, True)
    if not raw_output:
        self.fail('Keypair has not been created!')