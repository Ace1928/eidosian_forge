import sys
import threading
from typing import Tuple
from wsgiref import simple_server
from dulwich.tests import SkipTest, skipIf
from ...server import DictBackend, ReceivePackHandler, UploadPackHandler
from ...web import (
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase
def patch_capabilities(handler, caps_removed):
    original_capabilities = handler.capabilities
    filtered_capabilities = [i for i in original_capabilities() if i not in caps_removed]

    def capabilities(cls):
        return filtered_capabilities
    handler.capabilities = classmethod(capabilities)
    return original_capabilities