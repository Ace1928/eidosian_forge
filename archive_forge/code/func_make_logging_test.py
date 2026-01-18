import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def make_logging_test(**kwargs):

    def wrapper(fn):

        def test_fn(self):
            torch._dynamo.reset()
            records = []
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)
        return test_fn
    return wrapper