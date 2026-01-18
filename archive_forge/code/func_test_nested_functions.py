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
def test_nested_functions(self):

    async def func(value):
        value = await sub_func(value * 2)
        return value * 3

    async def sub_func(value):
        value = await duet.completed_future(value * 5)
        return value * 7
    assert duet.run(func, 1) == 2 * 3 * 5 * 7