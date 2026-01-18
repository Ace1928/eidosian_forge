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
def test_failed_nested_generator(self):
    side_effects = []

    async def func(value):
        try:
            await sub_func(value * 2)
            return value * 3
        except Exception:
            return value * 5

    async def sub_func(value):
        await duet.failed_future(Exception())
        side_effects.append(value * 7)
    assert duet.run(func, 1) == 5
    assert side_effects == []