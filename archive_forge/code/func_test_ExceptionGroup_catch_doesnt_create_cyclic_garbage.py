from __future__ import annotations
import gc
import sys
from traceback import extract_tb
from typing import TYPE_CHECKING, Callable, NoReturn
import pytest
from .._concat_tb import concat_tb
@pytest.mark.skipif(sys.implementation.name != 'cpython', reason='Only makes sense with refcounting GC')
def test_ExceptionGroup_catch_doesnt_create_cyclic_garbage() -> None:
    gc.collect()
    old_flags = gc.get_debug()

    def make_multi() -> NoReturn:
        raise ExceptionGroup('', [get_exc(raiser1), get_exc(raiser2)])
    try:
        gc.set_debug(gc.DEBUG_SAVEALL)
        with pytest.raises(ExceptionGroup) as excinfo:
            raise make_multi()
        for exc in excinfo.value.exceptions:
            assert isinstance(exc, (ValueError, KeyError))
        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()