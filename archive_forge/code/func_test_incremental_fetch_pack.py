import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def test_incremental_fetch_pack(self):
    self.test_fetch_pack()
    dest, dummy = self.disable_ff_and_make_dummy_commit()
    dest.refs[b'refs/heads/master'] = dummy
    c = self._client()
    repo_dir = os.path.join(self.gitroot, 'server_new.export')
    with repo.Repo(repo_dir) as dest:
        result = c.fetch(self._build_path('/dest'), dest)
        for r in result.refs.items():
            dest.refs.set_if_equals(r[0], None, r[1])
        self.assertDestEqualsSrc()