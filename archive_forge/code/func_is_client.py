import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
@property
def is_client(self):
    """ (Read-only) Is this a client instance? """
    return self._wsproto.client