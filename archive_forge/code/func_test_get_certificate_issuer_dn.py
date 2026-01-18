import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_get_certificate_issuer_dn(self):
    root_cert, root_key = unit.create_certificate(unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organization_name='fujitsu', organizational_unit_name='test', common_name='root'))
    cert_pem = unit.create_pem_certificate(unit.create_dn(common_name='test', organization_name='dev', locality_name='suzhou', state_or_province_name='jiangsu', country_name='cn', user_id='user_id', domain_component='test.com', email_address='user@test.com'), ca=root_cert, ca_key=root_key)
    dn = common_utils.get_certificate_subject_dn(cert_pem)
    self.assertEqual('test', dn.get('CN'))
    self.assertEqual('dev', dn.get('O'))
    self.assertEqual('suzhou', dn.get('L'))
    self.assertEqual('jiangsu', dn.get('ST'))
    self.assertEqual('cn', dn.get('C'))
    self.assertEqual('user_id', dn.get('UID'))
    self.assertEqual('test.com', dn.get('DC'))
    self.assertEqual('user@test.com', dn.get('emailAddress'))
    dn = common_utils.get_certificate_issuer_dn(cert_pem)
    self.assertEqual('root', dn.get('CN'))
    self.assertEqual('fujitsu', dn.get('O'))
    self.assertEqual('kawasaki', dn.get('L'))
    self.assertEqual('kanagawa', dn.get('ST'))
    self.assertEqual('jp', dn.get('C'))
    self.assertEqual('test', dn.get('OU'))