import copy
import os
import socket
import ssl
import sys
import threading
import weakref
from abc import abstractmethod
from itertools import chain
from queue import Empty, Full, LifoQueue
from time import time
from typing import Any, Callable, List, Optional, Type, Union
from urllib.parse import parse_qs, unquote, urlparse
from ._parsers import Encoder, _HiredisParser, _RESP2Parser, _RESP3Parser
from .backoff import NoBackoff
from .credentials import CredentialProvider, UsernamePasswordCredentialProvider
from .exceptions import (
from .retry import Retry
from .utils import (
def send_command(self, *args, **kwargs):
    """Pack and send a command to the Redis server"""
    self.send_packed_command(self._command_packer.pack(*args), check_health=kwargs.get('check_health', True))