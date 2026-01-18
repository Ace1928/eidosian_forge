import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
def test_fetch_from_dulwich_issue_88_alternative(self):
    self._source_repo = self.import_repo('issue88_expect_ack_nak_other.export')
    self._client_repo = self.import_repo('issue88_expect_ack_nak_client.export')
    port = self._start_server(self._source_repo)
    self.assertRaises(KeyError, self._client_repo.get_object, b'02a14da1fc1fc13389bbf32f0af7d8899f2b2323')
    run_git_or_fail(['fetch', self.url(port), 'master'], cwd=self._client_repo.path)
    self.assertEqual(b'commit', self._client_repo.get_object(b'02a14da1fc1fc13389bbf32f0af7d8899f2b2323').type_name)