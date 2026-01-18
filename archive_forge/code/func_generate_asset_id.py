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
def generate_asset_id(seed):
    rd = random.Random()
    rd.seed(seed)
    result = uuid.UUID(int=rd.getrandbits(128))
    return result