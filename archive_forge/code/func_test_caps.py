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
def test_caps(self):
    agent_cap = ('agent=dulwich/%d.%d.%d' % dulwich.__version__).encode('ascii')
    self.assertEqual({b'multi_ack', b'side-band-64k', b'ofs-delta', b'thin-pack', b'multi_ack_detailed', b'shallow', agent_cap}, set(self.client._fetch_capabilities))
    self.assertEqual({b'delete-refs', b'ofs-delta', b'report-status', b'side-band-64k', agent_cap}, set(self.client._send_capabilities))