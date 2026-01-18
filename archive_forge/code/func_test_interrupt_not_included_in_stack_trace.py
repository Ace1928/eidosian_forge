import abc
import concurrent.futures
import contextlib
import inspect
import sys
import time
import traceback
from typing import List, Tuple
import pytest
import duet
import duet.impl as impl
def test_interrupt_not_included_in_stack_trace(self):

    async def func():
        async with duet.new_scope() as scope:
            f = duet.AwaitableFuture()
            scope.spawn(lambda: f)
            f.set_exception(ValueError('oops!'))
            await duet.AwaitableFuture()
    with pytest.raises(ValueError, match='oops!') as exc_info:
        duet.run(func)
    stack_trace = ''.join(traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb))
    assert 'Interrupt' not in stack_trace
    assert isinstance(exc_info.value.__context__, impl.Interrupt)
    assert exc_info.value.__suppress_context__