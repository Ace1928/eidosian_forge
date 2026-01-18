from oslo_config import fixture as cfg_fixture
from oslo_messaging import conffixture as msg_fixture
from oslotest import createfile
import webob.dec
from keystonemiddleware import audit
from keystonemiddleware.tests.unit import utils
@property
def audit_map(self):
    return self.audit_map_file_fixture.path