import unittest
from unittest import mock
def test_integration_with_spec_method_definition_respects_spec(self):
    """You cannot define methods out of the spec"""
    m = mock.Mock(SampleObject)
    with self.assertRaises(AttributeError):
        m.method_sample3.return_value = 3