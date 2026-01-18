from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def test_interdependent_asyncgen_cleanup_order() -> None:
    saved: list[AsyncGenerator[int, None]] = []
    record: list[int | str] = []

    async def innermost() -> AsyncGenerator[int, None]:
        try:
            yield 1
        finally:
            await _core.cancel_shielded_checkpoint()
            record.append('innermost')

    async def agen(label: int, inner: AsyncGenerator[int, None]) -> AsyncGenerator[int, None]:
        try:
            yield (await inner.asend(None))
        finally:
            with pytest.raises(StopAsyncIteration):
                await inner.asend(None)
            record.append(label)

    async def async_main() -> None:
        ag_chain = innermost()
        for idx in range(100):
            ag_chain = agen(idx, ag_chain)
        saved.append(ag_chain)
        assert await ag_chain.asend(None) == 1
        assert record == []
    _core.run(async_main)
    assert record == ['innermost', *range(100)]