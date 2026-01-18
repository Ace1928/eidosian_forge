from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import posixpath
from xml.dom.minidom import parseString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import CorsTranslation
def test_cors_doc_not_wrapped_in_json_list(self):
    bucket_uri = self.CreateBucket()
    fpath = self.CreateTempFile(contents=self.cors_doc_not_nested_in_list.encode(UTF8))
    stderr = self.RunGsUtil(self._set_cmd_prefix + [fpath, suri(bucket_uri)], expected_status=1, return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn("'str' object has no attribute 'items'", stderr)
    else:
        self.assertIn('should be formatted as a list', stderr)