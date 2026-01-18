import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
@keylog_filename.setter
def keylog_filename(self, value: str) -> None:
    self._ctx.keylog_filename = value