from heat_integrationtests.functional import functional_base
def assert_empty_sections(self, changes, empty_sections):
    for section in empty_sections:
        self.assertEqual([], changes[section])