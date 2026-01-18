from cinderclient.contrib import noauth
from cinderclient.tests.unit import utils
def test_get_headers(self):
    headers = {'x-user-id': 'user', 'x-project-id': 'project', 'X-Auth-Token': 'user:project'}
    self.assertEqual(headers, self.plugin.get_headers(None))