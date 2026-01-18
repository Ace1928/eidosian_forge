import sys
import threading
from typing import Tuple
from wsgiref import simple_server
from dulwich.tests import SkipTest, skipIf
from ...server import DictBackend, ReceivePackHandler, UploadPackHandler
from ...web import (
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase
def test_new_shallow_clone_from_dulwich(self):
    raise SkipTest('Dumb web shallow cloning not supported.')