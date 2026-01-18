import asyncio
import signal
import socket
import warnings
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional, Set
from yarl import URL
from .typedefs import PathLike
from .web_app import Application
from .web_server import Server
@property
def addresses(self) -> List[Any]:
    ret: List[Any] = []
    for site in self._sites:
        server = site._server
        if server is not None:
            sockets = server.sockets
            if sockets is not None:
                for sock in sockets:
                    ret.append(sock.getsockname())
    return ret