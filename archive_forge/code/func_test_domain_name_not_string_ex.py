from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_domain_name_not_string_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'domain': {'name': {}}}}
    self._expect_failure(p)