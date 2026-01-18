import unittest
from unittest import mock
def test_integration_with_spec_method_definition(self):
    """You need to define the methods, even if they are in the spec"""
    m = mock.Mock(SampleObject)
    m.method_sample1.return_value = 1
    mock.seal(m)
    self.assertEqual(m.method_sample1(), 1)
    with self.assertRaises(AttributeError):
        m.method_sample2()