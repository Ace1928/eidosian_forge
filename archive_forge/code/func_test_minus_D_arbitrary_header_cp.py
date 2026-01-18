from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import platform
import re
import six
import gslib
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.utils.unit_util import ONE_KIB
def test_minus_D_arbitrary_header_cp(self):
    """Test upload and download with a sample perf trace token."""
    file_name = 'bar'
    fpath = self.CreateTempFile(file_name=file_name, contents=b'foo')
    bucket_uri = self.CreateBucket()
    stderr = self.RunGsUtil(['-D', '-h', 'custom-header:asdf', 'cp', fpath, suri(bucket_uri)], return_stderr=True)
    self.assertRegex(stderr, "Headers: \\{[\\s\\S]*'custom-header': 'asdf'[\\s\\S]*\\}")
    stderr2 = self.RunGsUtil(['-D', '-h', 'custom-header:asdf', 'cp', suri(bucket_uri, file_name), fpath], return_stderr=True)
    self.assertRegex(stderr2, "Headers: \\{[\\s\\S]*'custom-header': 'asdf'[\\s\\S]*\\}")
    stdout = self.RunGsUtil(['ls', '-L', suri(bucket_uri, file_name)], return_stdout=True)
    self.assertNotIn('custom', stdout)
    self.assertNotIn('asdf', stdout)