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
def assert_header_in_output(self, name, value, output):
    """Asserts that httplib2's debug logger printed out a specified header.

    This method is fairly primitive and uses assertIn statements, and thus is
    case-sensitive. Values should be normalized (e.g. to lowercase) if
    capitalization of the expected characters may vary.

    Args:
      name: (str) The header name, e.g. "Content-Length".
      value: (Union[str, None]) The header value, e.g. "4096". If no value is
          expected for the header or the value is unknown, this argument should
          be `None`.
      output: (str) The string in which to search for the specified header.
    """
    expected = 'header: %s:' % name
    if value:
        expected += ' %s' % value
    if expected in output:
        return
    alt_expected = "('%s'" % name
    if value:
        alt_expected += ", '%s')" % value
    if not alt_expected in output:
        self.fail('Neither of these two header formats were found in the output:\n1) %s\n2) %s\nOutput string: %s' % (expected, alt_expected, output))