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
def test_fetch_from_dulwich_issue_88_standard(self):
    self._source_repo = self.import_repo('issue88_expect_ack_nak_server.export')
    self._client_repo = self.import_repo('issue88_expect_ack_nak_client.export')
    port = self._start_server(self._source_repo)
    run_git_or_fail(['fetch', self.url(port), 'master'], cwd=self._client_repo.path)
    self.assertObjectStoreEqual(self._source_repo.object_store, self._client_repo.object_store)