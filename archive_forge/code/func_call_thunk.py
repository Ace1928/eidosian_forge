import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def call_thunk(self):
    """Call a vectorized thunk.

        Thunks have no arguments and can thus be called directly.

        """
    if self.is_coroutine_fn:
        loop = asyncio.new_event_loop()
        try:
            outputs = loop.run_until_complete(self.func())
        finally:
            loop.close()
    else:
        outputs = self.func()
    return outputs