import requests
import testtools.matchers
from keystone.tests.functional import core as functests
def test_admin_versions(self):
    for version in versions:
        resp = requests.get(self.ADMIN_URL + '/' + version)
        self.assertThat(resp.status_code, testtools.matchers.Annotate('failed for version %s' % version, is_ok))