import copy
import testtools
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import certificates
def test_create_fail(self):
    create_cert_fail = copy.deepcopy(CREATE_CERT)
    create_cert_fail['wrong_key'] = 'wrong'
    self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(certificates.CREATION_ATTRIBUTES), self.mgr.create, **create_cert_fail)
    self.assertEqual([], self.api.calls)