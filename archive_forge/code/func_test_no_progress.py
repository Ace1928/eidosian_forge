import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
def test_no_progress(self):
    caps = [*list(self._handler.required_capabilities()), b'no-progress']
    self._handler.set_client_capabilities(caps)
    self._handler.progress(b'first message')
    self._handler.progress(b'second message')
    self.assertRaises(IndexError, self._handler.proto.get_received_line, 2)