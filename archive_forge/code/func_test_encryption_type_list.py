from cinderclient.tests.functional import base
def test_encryption_type_list(self):
    encrypt_list = self.cinder('encryption-type-list')
    self.assertTableHeaders(encrypt_list, ['Volume Type ID', 'Provider', 'Cipher', 'Key Size', 'Control Location'])