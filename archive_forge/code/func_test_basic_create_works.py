import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_basic_create_works(self):
    """Make sure the working case is good.

        Note this combines test_override_aws_ec2_instance into this test as
        well, which is:
        If AWS::EC2::Instance is overridden, InstanceGroup will automatically
        use that overridden resource type.
        """
    files = {'provider.yaml': self.instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 4, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'OS::Heat::InstanceGroup'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    stack = self.client.stacks.get(stack_identifier)
    self.assert_instance_count(stack, 4)