from heat_integrationtests.functional import functional_base
def test_cancel_update_server_with_port(self):
    parameters = {'InstanceType': self.conf.minimal_instance_type, 'ImageId': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
    stack_identifier = self.stack_create(template=self.template, parameters=parameters)
    parameters['InstanceType'] = self.conf.instance_type
    self.update_stack(stack_identifier, self.template, parameters=parameters, expected_status='UPDATE_IN_PROGRESS')
    self._wait_for_resource_status(stack_identifier, 'Server', 'CREATE_IN_PROGRESS')
    self.cancel_update_stack(stack_identifier)