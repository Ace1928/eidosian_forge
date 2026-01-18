from boto.compat import StringIO
from tests.compat import mock, unittest
from boto.glacier import vault
from boto.glacier.job import Job
from boto.glacier.response import GlacierResponse
def test_large_part_size_is_obeyed(self):
    self.vault.DefaultPartSize = 8 * 1024 * 1024
    self.vault.create_archive_writer = mock.Mock()
    self.getsize.return_value = 1
    with mock.patch('boto.glacier.vault.open', self.mock_open, create=True):
        self.vault.create_archive_from_file('myfile')
    self.vault.create_archive_writer.assert_called_with(description=mock.ANY, part_size=self.vault.DefaultPartSize)