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
def assertDestEqualsSrc(self):
    repo_dir = os.path.join(self.gitroot, 'server_new.export')
    dest_repo_dir = os.path.join(self.gitroot, 'dest')
    with repo.Repo(repo_dir) as src:
        with repo.Repo(dest_repo_dir) as dest:
            self.assertReposEqual(src, dest)