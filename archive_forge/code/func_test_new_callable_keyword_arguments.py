import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_keyword_arguments(self):

    class Bar(object):
        kwargs = None

        def __init__(self, **kwargs):
            Bar.kwargs = kwargs
    patcher = patch(foo_name, new_callable=Bar, arg1=1, arg2=2)
    m = patcher.start()
    try:
        self.assertIs(type(m), Bar)
        self.assertEqual(Bar.kwargs, dict(arg1=1, arg2=2))
    finally:
        patcher.stop()