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
def send_and_verify(self, branch, local, target):
    """Send branch from local to remote repository and verify it worked."""
    client = LocalGitClient()
    ref_name = b'refs/heads/' + branch
    result = client.send_pack(target.path, lambda _: {ref_name: local.refs[ref_name]}, local.generate_pack_data)
    self.assertEqual(local.refs[ref_name], result.refs[ref_name])
    self.assertIs(None, result.agent)
    self.assertEqual({}, result.ref_status)
    obj_local = local.get_object(result.refs[ref_name])
    obj_target = target.get_object(result.refs[ref_name])
    self.assertEqual(obj_local, obj_target)