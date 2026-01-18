from unittest import mock
from heat.engine import attributes
from heat.engine import resources
from heat.engine import support
from heat.tests import common
def test_caching_none(self):
    self.resolver.side_effect = ['value3', 'value3 changed']
    attribs = attributes.Attributes('test resource', self.attributes_schema, self.resolver)
    self.assertEqual('value3', attribs['test3'])
    self.assertEqual('value3 changed', attribs['test3'])
    calls = [mock.call('test3'), mock.call('test3')]
    self.resolver.assert_has_calls(calls)