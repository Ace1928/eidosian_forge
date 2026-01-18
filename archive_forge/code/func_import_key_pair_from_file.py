import os
import re
import time
import atexit
import random
import socket
import hashlib
import binascii
import datetime
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Type, Tuple, Union, Callable, Optional
import libcloud.compute.ssh
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import b
from libcloud.common.base import BaseDriver, Connection, ConnectionKey
from libcloud.compute.ssh import SSHClient, BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.common.types import LibcloudError
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet, is_valid_ip_address
def import_key_pair_from_file(self, name, key_file_path):
    """
        Import a new public key from string.

        :param name: Key pair name.
        :type name: ``str``

        :param key_file_path: Path to the public key file.
        :type key_file_path: ``str``

        :rtype: :class:`.KeyPair` object
        """
    key_file_path = os.path.expanduser(key_file_path)
    with open(key_file_path) as fp:
        key_material = fp.read().strip()
    return self.import_key_pair_from_string(name=name, key_material=key_material)