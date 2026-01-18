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
def register_vendor(self, name, vendor):
    """Register new SSH vendor by name."""
    self._ssh_vendors[name] = vendor