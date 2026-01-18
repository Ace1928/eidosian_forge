from io import BytesIO
from unittest import TestCase
from fastimport import (
from fastimport.processors import (
from :2
from :2
from :100
from :101
from :100
from :100
from :100
from :100
from :101
from :100
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :102
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :100
from :100
from :100
from :100
from :102
from :101
from :102
from :101
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
def test_deleteall(self):
    params = {b'include_paths': [b'doc/index.txt']}
    self.assertFiltering(_SAMPLE_WITH_DELETEALL, params, b'blob\nmark :4\ndata 11\n== Docs ==\ncommit refs/heads/master\nmark :102\ncommitter d <b@c> 1234798653 +0000\ndata 8\ntest\ning\nfrom :100\ndeleteall\nM 644 :4 index.txt\n')