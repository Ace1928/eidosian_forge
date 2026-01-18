from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def back_in_main() -> tuple[str, str]:
    back_parent_value = trio_test_contextvar.get()
    trio_test_contextvar.set('back_in_main')
    back_current_value = trio_test_contextvar.get()
    assert sniffio.current_async_library() == 'trio'
    return (back_parent_value, back_current_value)