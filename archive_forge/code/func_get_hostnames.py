import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def get_hostnames(self):
    """
        Return the set of literal hostnames defined in the SSH config (both
        explicit hostnames and wildcard entries).
        """
    hosts = set()
    for entry in self._config:
        hosts.update(entry['host'])
    return hosts