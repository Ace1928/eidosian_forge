import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def make_settings_test(settings):

    def wrapper(fn):

        def test_fn(self):
            torch._dynamo.reset()
            records = []
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)
        return test_fn
    return wrapper