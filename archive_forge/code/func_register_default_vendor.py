import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def register_default_vendor(self, vendor):
    """Register default SSH vendor."""
    self._default_ssh_vendor = vendor