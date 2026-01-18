import io
import sys
import requests
import testtools
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient.tests import utils as test_utils
def test_iter_iterator_display_progress_bar(self):
    size = 100
    resp = requests.Response()
    resp.headers['x-openstack-request-id'] = 'req-1234'
    iterator_with_len = utils.IterableWithLength(iter('X' * 100), size)
    requestid_proxy = utils.RequestIdProxy((iterator_with_len, resp))
    saved_stdout = sys.stdout
    try:
        sys.stdout = output = test_utils.FakeTTYStdout()
        data = list(progressbar.VerboseIteratorWrapper(requestid_proxy, size))
        self.assertEqual(['X'] * 100, data)
        self.assertEqual('[%s>] 100%%\n' % ('=' * 29), output.getvalue())
    finally:
        sys.stdout = saved_stdout