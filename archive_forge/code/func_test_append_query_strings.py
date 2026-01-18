import testtools
from unittest import mock
from troveclient import common
def test_append_query_strings(self):
    url = 'test-url'
    self.assertEqual(url, common.append_query_strings(url))
    self.assertEqual(url, common.append_query_strings(url, limit=None, marker=None))
    limit = 'test-limit'
    marker = 'test-marker'
    result = common.append_query_strings(url, limit=limit, marker=marker)
    self.assertTrue(result.startswith(url + '?'))
    self.assertIn('limit=%s' % limit, result)
    self.assertIn('marker=%s' % marker, result)
    self.assertEqual(1, result.count('&'))
    opts = {}
    self.assertEqual(url, common.append_query_strings(url, limit=None, marker=None, **opts))
    opts = {'key1': 'value1', 'key2': None}
    result = common.append_query_strings(url, limit=None, marker=marker, **opts)
    self.assertTrue(result.startswith(url + '?'))
    self.assertEqual(1, result.count('&'))
    self.assertNotIn('limit=%s' % limit, result)
    self.assertIn('marker=%s' % marker, result)
    self.assertIn('key1=%s' % opts['key1'], result)
    self.assertNotIn('key2=%s' % opts['key2'], result)
    opts = {'key1': 'value1', 'key2': None, 'key3': False}
    result = common.append_query_strings(url, **opts)
    self.assertTrue(result.startswith(url + '?'))
    self.assertIn('key1=value1', result)
    self.assertNotIn('key2', result)
    self.assertIn('key3=False', result)