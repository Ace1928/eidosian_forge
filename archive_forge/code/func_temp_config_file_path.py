import os.path
import fixtures
from oslo_config import cfg
import testtools
def temp_config_file_path(self, name='api_audit_map.conf'):
    return os.path.join(self.tempdir.path, name)