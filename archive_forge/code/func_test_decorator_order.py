from boto.mws.connection import MWSConnection, api_call_map, destructure_object
from boto.mws.response import (ResponseElement, GetFeedSubmissionListResult,
from boto.exception import BotoServerError
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from mock import MagicMock
def test_decorator_order(self):
    for action, func in api_call_map.items():
        func = getattr(self.service_connection, func)
        decs = [func.__name__]
        while func:
            i = 0
            if not hasattr(func, '__closure__'):
                func = getattr(func, '__wrapped__', None)
                continue
            while i < len(func.__closure__):
                value = func.__closure__[i].cell_contents
                if hasattr(value, '__call__'):
                    if 'requires' == value.__name__:
                        self.assertTrue(not decs or decs[-1] == 'requires')
                    decs.append(value.__name__)
                i += 1
            func = getattr(func, '__wrapped__', None)