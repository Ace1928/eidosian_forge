import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_no_infinite_recursion(self):
    """Prove that we can override a python resource.

        And use that resource within the template resource.
        """
    stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ)
    self.assert_resource_is_a_stack(stack_identifier, 'secret1')