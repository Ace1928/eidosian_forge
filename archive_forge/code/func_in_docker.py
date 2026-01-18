from __future__ import absolute_import, division, print_function
import json
import os
import sys
import uuid
import random
import re
import socket
from datetime import datetime
from ssl import SSLError
from http.client import RemoteDisconnected
from time import time
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import (
from .constants import (
from .version import CURRENT_COLL_VERSION
def in_docker():
    """Check to see if we are running in a container

    Returns:
        bool: True if in a container. False otherwise.
    """
    try:
        with open('/proc/1/cgroup') as fh:
            lines = fh.readlines()
    except IOError:
        return False
    if any(('/docker/' in x for x in lines)):
        return True
    return False