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
def serve_command(self, handler_cls, args, inf, outf):
    return serve_command(handler_cls, [b'test', *args], backend=self.backend, inf=inf, outf=outf)