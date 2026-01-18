import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
def test_connect(self):
    server = self.server
    client = self.client
    client.username = b'username'
    client.port = 1337
    client._connect(b'command', b'/path/to/repo')
    self.assertEqual(b'username', server.username)
    self.assertEqual(1337, server.port)
    self.assertEqual("git-command '/path/to/repo'", server.command)
    client._connect(b'relative-command', b'/~/path/to/repo')
    self.assertEqual("git-relative-command '~/path/to/repo'", server.command)