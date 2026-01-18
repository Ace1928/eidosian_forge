import unittest
import time
from nose.plugins.attrib import attr
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
from boto.s3.deletemarker import DeleteMarker
def test_mfadel(self):
    mfa_sn = raw_input('MFA S/N: ')
    mfa_code = raw_input('MFA Code: ')
    self.bucket.configure_versioning(True, mfa_delete=True, mfa_token=(mfa_sn, mfa_code))
    i = 0
    for i in range(1, 8):
        time.sleep(2 ** i)
        d = self.bucket.get_versioning_status()
        if d['Versioning'] == 'Enabled' and d['MfaDelete'] == 'Enabled':
            break
    self.assertEqual('Enabled', d['Versioning'])
    self.assertEqual('Enabled', d['MfaDelete'])
    k = self.bucket.new_key('foobar')
    s1 = 'This is v1'
    k.set_contents_from_string(s1)
    v1 = k.version_id
    try:
        self.bucket.delete_key('foobar', version_id=v1)
        self.fail('Must fail if not using MFA token')
    except S3ResponseError:
        pass
    mfa_code = raw_input('MFA Code: ')
    self.bucket.delete_key('foobar', version_id=v1, mfa_token=(mfa_sn, mfa_code))
    mfa_code = raw_input('MFA Code: ')
    self.bucket.configure_versioning(False, mfa_delete=False, mfa_token=(mfa_sn, mfa_code))
    i = 0
    for i in range(1, 8):
        time.sleep(2 ** i)
        d = self.bucket.get_versioning_status()
        if d['Versioning'] == 'Suspended' and d['MfaDelete'] != 'Enabled':
            break
    self.assertEqual('Suspended', d['Versioning'])
    self.assertNotEqual('Enabled', d['MfaDelete'])