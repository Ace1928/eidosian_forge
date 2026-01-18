from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
def test_non_admin_forbidden_create_resources(self):
    """Fail to create resource w/o admin role.

        Integration tests job runs as normal OpenStack user,
        and the resources above are configured to require
        admin role in default policy file of Heat.
        """
    if self.test_creation:
        ex = self.assertRaises(exc.Forbidden, self.client.stacks.create, stack_name=self.stack_name, template=self.template)
        self.assertIn(self.forbidden_r_type, ex.message.decode('utf-8'))