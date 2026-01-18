import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_mock_calls_with_patch(self):
    for arg in ('spec', 'autospec', 'spec_set'):
        p = patch('%s.SomeClass' % __name__, **{arg: True})
        m = p.start()
        try:
            m.wibble()
            kalls = [call.wibble()]
            self.assertEqual(m.mock_calls, kalls)
            self.assertEqual(m.method_calls, kalls)
            self.assertEqual(m.wibble.mock_calls, [call()])
            result = m()
            kalls.append(call())
            self.assertEqual(m.mock_calls, kalls)
            result.wibble()
            kalls.append(call().wibble())
            self.assertEqual(m.mock_calls, kalls)
            self.assertEqual(result.mock_calls, [call.wibble()])
            self.assertEqual(result.wibble.mock_calls, [call()])
            self.assertEqual(result.method_calls, [call.wibble()])
        finally:
            p.stop()