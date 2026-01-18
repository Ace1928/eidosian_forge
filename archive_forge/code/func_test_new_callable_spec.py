import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_spec(self):

    class Bar(object):
        kwargs = None

        def __init__(self, **kwargs):
            Bar.kwargs = kwargs
    patcher = patch(foo_name, new_callable=Bar, spec=Bar)
    patcher.start()
    try:
        self.assertEqual(Bar.kwargs, dict(spec=Bar))
    finally:
        patcher.stop()
    patcher = patch(foo_name, new_callable=Bar, spec_set=Bar)
    patcher.start()
    try:
        self.assertEqual(Bar.kwargs, dict(spec_set=Bar))
    finally:
        patcher.stop()