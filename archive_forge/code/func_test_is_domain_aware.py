import uuid
from keystone.common import driver_hints
from keystone import exception
def test_is_domain_aware(self):
    self.assertIs(self.expected_is_domain_aware, self.driver.is_domain_aware())