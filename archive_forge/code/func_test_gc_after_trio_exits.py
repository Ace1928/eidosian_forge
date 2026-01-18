from __future__ import annotations
import random
from contextlib import asynccontextmanager
from itertools import count
from typing import TYPE_CHECKING, NoReturn
import attrs
import pytest
from trio._tests.pytest_plugin import skip_if_optional_else_raise
import trio
import trio.testing
from trio import DTLSChannel, DTLSEndpoint
from trio.testing._fake_net import FakeNet, UDPPacket
from .._core._tests.tutil import binds_ipv6, gc_collect_harder, slow
@pytest.mark.filterwarnings('always:unclosed DTLS:ResourceWarning')
def test_gc_after_trio_exits() -> None:

    async def main() -> DTLSEndpoint:
        fn = FakeNet()
        fn.enable()
        return endpoint()
    e = trio.run(main)
    with pytest.warns(ResourceWarning):
        del e
        gc_collect_harder()