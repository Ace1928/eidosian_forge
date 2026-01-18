import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
def session_stats(self) -> dict[str, int]:
    return self._ctx.session_stats()