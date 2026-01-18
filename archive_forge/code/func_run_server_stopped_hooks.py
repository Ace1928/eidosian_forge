import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def run_server_stopped_hooks(self, backing_urls=None):
    if backing_urls is None:
        backing_urls = self._backing_urls()
    for hook in SmartTCPServer.hooks['server_stopped']:
        hook(backing_urls, self.get_url())