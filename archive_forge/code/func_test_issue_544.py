import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_issue_544(self):
    com = autocomplete.MultilineJediCompletion()
    code = '@asyncio.coroutine\ndef'
    history = ('import asyncio', '@asyncio.coroutin')
    com.matches(3, 'def', current_block=code, history=history)