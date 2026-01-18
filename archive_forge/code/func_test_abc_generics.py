from __future__ import annotations
import attrs
import pytest
from .. import abc as tabc
from ..lowlevel import Task
def test_abc_generics() -> None:

    class SlottedChannel(tabc.SendChannel[tabc.Stream]):
        __slots__ = ('x',)

        def send_nowait(self, value: object) -> None:
            raise RuntimeError

        async def send(self, value: object) -> None:
            raise RuntimeError

        def clone(self) -> None:
            raise RuntimeError

        async def aclose(self) -> None:
            pass
    channel = SlottedChannel()
    with pytest.raises(RuntimeError):
        channel.send_nowait(None)