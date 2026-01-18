import saharaclient
from saharaclient.api import base as api_base
from saharaclient.tests.unit import base
def test_get_query_string(self):
    res = api_base.get_query_string(None, limit=None, marker=None)
    self.assertEqual('', res)
    res = api_base.get_query_string(None, limit=4, marker=None)
    self.assertEqual('?limit=4', res)
    res = api_base.get_query_string({'opt1': 2}, limit=None, marker=3)
    self.assertEqual('?marker=3&opt1=2', res)