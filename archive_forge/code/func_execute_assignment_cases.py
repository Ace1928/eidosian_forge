from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def execute_assignment_cases(self, test_plan, test_data):
    """Execute the test plan, based on the created test_data."""

    def check_results(expected, actual, param_arg_count):
        if param_arg_count == 0:
            self.assertEqual(len(expected) + test_data['initial_assignment_count'], len(actual))
        else:
            self.assertThat(actual, matchers.HasLength(len(expected)))
        for each_expected in expected:
            expected_assignment = {}
            for param in each_expected:
                if param == 'inherited_to_projects':
                    expected_assignment[param] = each_expected[param]
                elif param == 'indirect':
                    indirect_term = {}
                    for indirect_param in each_expected[param]:
                        key, value = self._convert_entity_shorthand(indirect_param, each_expected[param], test_data)
                        indirect_term[key] = value
                    expected_assignment[param] = indirect_term
                else:
                    key, value = self._convert_entity_shorthand(param, each_expected, test_data)
                    expected_assignment[key] = value
            self.assertIn(expected_assignment, actual)

    def convert_group_ids_sourced_from_list(index_list, reference_data):
        value_list = []
        for group_index in index_list:
            value_list.append(reference_data['groups'][group_index]['id'])
        return value_list
    for test in test_plan.get('tests', []):
        args = {}
        for param in test['params']:
            if param in ['effective', 'inherited', 'include_subtree']:
                args[param] = test['params'][param]
            elif param == 'source_from_group_ids':
                args[param] = convert_group_ids_sourced_from_list(test['params']['source_from_group_ids'], test_data)
            else:
                key, value = self._convert_entity_shorthand(param, test['params'], test_data)
                args[key] = value
        results = PROVIDERS.assignment_api.list_role_assignments(**args)
        check_results(test['results'], results, len(args))