import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def get_vfat_transport(self, url):
    """Return vfat-backed transport for test directory"""
    from breezy.transport.fakevfat import FakeVFATTransportDecorator
    return FakeVFATTransportDecorator('vfat+' + url)