from __future__ import annotations
import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any
import trio
from trio.socket import SOCK_STREAM, SocketType, getaddrinfo, socket
def reorder_for_rfc_6555_section_5_4(targets: list[tuple[AddressFamily, SocketKind, int, str, Any]]) -> None:
    for i in range(1, len(targets)):
        if targets[i][0] != targets[0][0]:
            if i != 1:
                targets.insert(1, targets.pop(i))
            break