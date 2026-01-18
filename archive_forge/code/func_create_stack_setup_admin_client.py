from heat_integrationtests.functional import functional_base
def create_stack_setup_admin_client(self, template=test_template):
    self.stack_identifier = self.stack_create(template=template)
    self.setup_clients_for_admin()