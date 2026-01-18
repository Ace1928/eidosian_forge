import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_config_invalid_domain(self):
    """Call ``PUT /domains/{domain_id}/config``.

        While creating Identity API-based domain config with an invalid domain
        id provided, the request shall be rejected with a response, 404 domain
        not found.
        """
    invalid_domain_id = uuid.uuid4().hex
    url = '/domains/%(domain_id)s/config' % {'domain_id': invalid_domain_id}
    self.put(url, body={'config': self.config}, expected_status=exception.DomainNotFound.code)