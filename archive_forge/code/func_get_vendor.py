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
def get_vendor(self):
    """Find out what version of SSH is on the system.

        :raises SSHVendorNotFound: if no any SSH vendor is found
        :raises UnknownSSH: if the BRZ_SSH environment variable contains
                            unknown vendor name
        """
    if self._cached_ssh_vendor is None:
        vendor = self._get_vendor_by_config()
        if vendor is None:
            vendor = self._get_vendor_by_inspection()
            if vendor is None:
                trace.mutter('falling back to default implementation')
                vendor = self._default_ssh_vendor
                if vendor is None:
                    raise errors.SSHVendorNotFound()
        self._cached_ssh_vendor = vendor
    return self._cached_ssh_vendor