from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier.job import Job
from boto.glacier.layer1 import Layer1
from boto.glacier.response import GlacierResponse
from boto.glacier.exceptions import TreeHashDoesNotMatchError
def test_get_job_validate_checksum_success(self):
    response = GlacierResponse(mock.Mock(), None)
    response['TreeHash'] = 'tree_hash'
    self.api.get_job_output.return_value = response
    with mock.patch('boto.glacier.job.tree_hash_from_str') as t:
        t.return_value = 'tree_hash'
        self.job.get_output(byte_range=(1, 1024), validate_checksum=True)