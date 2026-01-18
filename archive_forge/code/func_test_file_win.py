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
@patch('os.name', 'nt')
@patch('sys.platform', 'win32')
def test_file_win(self):
    from nturl2path import url2pathname
    with patch('dulwich.client.url2pathname', url2pathname):
        expected = 'C:\\foo.bar\\baz'
        for file_url in ['file:C:/foo.bar/baz', 'file:/C:/foo.bar/baz', 'file://C:/foo.bar/baz', 'file://C://foo.bar//baz', 'file:///C:/foo.bar/baz']:
            c, path = get_transport_and_path(file_url)
            self.assertIsInstance(c, LocalGitClient)
            self.assertEqual(path, expected)
        for remote_url in ['file://host.example.com/C:/foo.bar/bazfile://host.example.com/C:/foo.bar/bazfile:////host.example/foo.bar/baz']:
            with self.assertRaises(NotImplementedError):
                c, path = get_transport_and_path(remote_url)