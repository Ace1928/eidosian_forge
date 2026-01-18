import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_by_option_invalid_domain(self):
    """Call ``GET & HEAD /domains{domain_id}/config/{group}/{option}``.

        While retrieving Identity API-based domain config by option with an
        invalid domain id provided, the request shall be rejected with a
        response 404 domain not found.
        """
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_domain_id = uuid.uuid4().hex
    url = '/domains/%(domain_id)s/config/ldap/url' % {'domain_id': invalid_domain_id}
    self.get(url, expected_status=exception.DomainNotFound.code)
    self.head(url, expected_status=exception.DomainNotFound.code)