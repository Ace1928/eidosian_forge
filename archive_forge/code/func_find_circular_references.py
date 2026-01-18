import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
def find_circular_references(garbage):
    """Find circular references in a list of objects.

    The garbage list contains objects that participate in a cycle,
    but also the larger set of objects kept alive by that cycle.
    This function finds subsets of those objects that make up
    the cycle(s).
    """

    def inner(level):
        for item in level:
            item_id = id(item)
            if item_id not in garbage_ids:
                continue
            if item_id in visited_ids:
                continue
            if item_id in stack_ids:
                candidate = stack[stack.index(item):]
                candidate.append(item)
                found.append(candidate)
                continue
            stack.append(item)
            stack_ids.add(item_id)
            inner(gc.get_referents(item))
            stack.pop()
            stack_ids.remove(item_id)
            visited_ids.add(item_id)
    found: typing.List[object] = []
    stack = []
    stack_ids = set()
    garbage_ids = set(map(id, garbage))
    visited_ids = set()
    inner(garbage)
    return found