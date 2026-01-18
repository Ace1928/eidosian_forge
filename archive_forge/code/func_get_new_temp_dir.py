import fixtures
from oslo_config import fixture as config
from oslotest import base as test_base
from oslo_service import _options
from oslo_service import sslutils
def get_new_temp_dir(self):
    """Create a new temporary directory.

        :returns: fixtures.TempDir
        """
    return self.useFixture(fixtures.TempDir())