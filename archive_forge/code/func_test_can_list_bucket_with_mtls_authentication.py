from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto import config
from gslib.tests import testcase
from gslib.tests.testcase import integration_testcase
from gslib.tests.util import unittest
@unittest.skipIf(not config.getbool('Credentials', 'use_client_certificate'), 'mTLS requires "use_client_certificate" to be "True" in .boto config.')
@integration_testcase.SkipForXML(MTLS_AVAILABILITY_MESSAGE)
@integration_testcase.SkipForS3(MTLS_AVAILABILITY_MESSAGE)
def test_can_list_bucket_with_mtls_authentication(self):
    bucket_uri = 'gs://{}'.format(self.MakeTempName('bucket'))
    self.RunGsUtil(['mb', bucket_uri])
    stdout = self.RunGsUtil(['-D', 'ls'], return_stdout=True)
    self.RunGsUtil(['rb', bucket_uri])
    self.assertIn('storage.mtls.googleapis.com', stdout)
    self.assertIn(bucket_uri, stdout)