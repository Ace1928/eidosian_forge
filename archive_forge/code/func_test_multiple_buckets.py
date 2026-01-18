from __future__ import absolute_import
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
@SkipForXML('Public access prevention only runs on GCS JSON API')
def test_multiple_buckets(self):
    bucket_uri1 = self.CreateBucket()
    bucket_uri2 = self.CreateBucket()
    stdout = self.RunGsUtil(self._get_pap_cmd + [suri(bucket_uri1), suri(bucket_uri2)], return_stdout=True)
    self.assertRegex(stdout, '%s:\\s+inherited' % suri(bucket_uri1))
    self.assertRegex(stdout, '%s:\\s+inherited' % suri(bucket_uri2))