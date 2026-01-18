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
def test_init_username_set_no_password(self):
    url = 'https://github.com/jelmer/dulwich'
    c = HttpGitClient(url, config=None, username='user')
    self.assertEqual('user', c._username)
    self.assertIsNone(c._password)
    basic_auth = c.pool_manager.headers['authorization']
    auth_string = b'user:'
    b64_credentials = base64.b64encode(auth_string)
    expected_basic_auth = f'Basic {b64_credentials.decode('ascii')}'
    self.assertEqual(basic_auth, expected_basic_auth)