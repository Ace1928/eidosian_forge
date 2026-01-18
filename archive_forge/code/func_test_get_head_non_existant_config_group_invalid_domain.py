import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_non_existant_config_group_invalid_domain(self):
    """Call ``GET & HEAD /domains/{domain_id}/config/{group}``.

        While retrieving non-existent Identity API-based domain config group
        with an invalid domain id provided, the request shall be rejected with
        a response, 404 domain not found.
        """
    config = {'ldap': {'url': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    invalid_domain_id = uuid.uuid4().hex
    url = '/domains/%(domain_id)s/config/identity' % {'domain_id': invalid_domain_id}
    self.get(url, expected_status=exception.DomainNotFound.code)
    self.head(url, expected_status=exception.DomainNotFound.code)