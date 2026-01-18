from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def mock_list(arg_url, arg_response_key):
    try:
        result = self.results[self.result_index]
    except IndexError:
        return []
    self.result_index = self.result_index + 1
    limit_string = 'limit=%s' % self.limit
    self.assertIn(limit_string, arg_url)
    offset = result[0]
    if offset > 0:
        offset_string = 'marker=abcd1234-%s' % offset
        self.assertIn(offset_string, arg_url)

    def results():
        for i in range(*result):
            self.limit -= 1
            stack_name = 'stack_%s' % (i + 1)
            stack_id = 'abcd1234-%s' % (i + 1)
            yield mock_stack(manager, stack_name, stack_id)
    return list(results())