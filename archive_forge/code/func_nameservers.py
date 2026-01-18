import asyncio
import functools
import pycares
import socket
import sys
from typing import (
from . import error
@nameservers.setter
def nameservers(self, value: Sequence[str]) -> None:
    self._channel.servers = value