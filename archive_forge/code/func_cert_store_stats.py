import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
def cert_store_stats(self) -> dict[str, int]:
    raise NotImplementedError()