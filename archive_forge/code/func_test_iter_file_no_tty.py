import io
import sys
import requests
import testtools
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient.tests import utils as test_utils
def test_iter_file_no_tty(self):
    size = 98304
    file_obj = io.StringIO('X' * size)
    saved_stdout = sys.stdout
    try:
        sys.stdout = output = test_utils.FakeNoTTYStdout()
        file_obj = progressbar.VerboseFileWrapper(file_obj, size)
        chunksize = 1024
        chunk = file_obj.read(chunksize)
        while chunk:
            chunk = file_obj.read(chunksize)
        self.assertEqual('', output.getvalue())
    finally:
        sys.stdout = saved_stdout